import numpy as np


class ImpellerFlowModel:
    """
    Solver-agnostic impeller flow model.

    All solver-specific parameter names are isolated in PARAMETER_MAP.
    """

    # ============================================================
    # Solver parameter mapping (ONLY place names appear)
    # ============================================================

    PARAMETER_MAP = {
        "inlet_relative_velocity_tip": "W1T",
        "inlet_blade_speed_tip": "U1T",
        ... # Add other parameters as needed
    }

    def __init__(self, solver, lib=np):
        self.solver = solver
        self.lib = lib
        self.W2s = 0.0

    # ============================================================
    # Parameter access
    # ============================================================

    def get_known_parameters(self):
        """
        Pull required parameters from solver into semantic attributes.
        """
        for attr, solver_key in self.PARAMETER_MAP.items():
            setattr(self, attr, self.solver.GetParameter(1, solver_key))

    # ============================================================
    # Primary zone (rothalpy)
    # ============================================================

    def solve_primary_zone_rothalpy(self):
        rhs = (
            self.specific_heat * self.inlet_total_temperature_tip
            + 0.5 * (
                self.inlet_relative_velocity_tip**2
                - self.inlet_blade_speed_tip**2
                + self.exit_blade_speed**2
            )
        )

        T2p_coeff = (
            self.specific_heat
            + 0.5
            * self.inlet_relative_velocity_tip**2
            / (self.MR2**2 * self.inlet_total_temperature_tip)
        )

        self.T2p = rhs / T2p_coeff
        self.p2 = self.inlet_total_pressure_tip * (
            self.T2p / self.inlet_total_temperature_tip
        ) ** (self.gamma / (self.gamma - 1))

        self.W2p = (
            self.inlet_relative_velocity_tip / self.MR2
        ) * self.lib.sqrt(self.T2p / self.inlet_total_temperature_tip)

        self.hT = (
            self.specific_heat * self.T2p
            + 0.5 * (self.W2p**2 - self.exit_blade_speed**2)
        )

        self.rho2p = self.p2 / (self.gas_constant * self.T2p)

    def make_MR2_objective(self, p_ref):
        self.p_ref = p_ref

        def objective(MR2):
            self.MR2 = MR2[0]
            self.solve_primary_zone_rothalpy()
            return self.p2 - self.p_ref

        return objective

    # ============================================================
    # Velocity calculations
    # ============================================================

    def calculate_primary_zone_velocities(self):
        self.beta_2p = self.lib.radians(self.exit_blade_angle + self.delta_2p)
        self.Cm2p = self.W2p * self.lib.cos(self.beta_2p)
        self.Ctheta2p = self.exit_blade_speed + self.Cm2p * self.lib.tan(self.beta_2p)
        self.C2p = self.lib.sqrt(self.Cm2p**2 + self.Ctheta2p**2)

    def calculate_epsilon(self):
        Ageo = (
            2 * self.lib.pi * self.exit_radius * self.exit_width
            - self.blade_count
            * self.exit_width
            * self.trailing_edge_thickness
            / self.lib.cos(self.lib.radians(self.exit_blade_angle))
        )

        self.eps = 1 - self.mass_flow * (1 - self.chi) / (
            self.rho2p * self.Cm2p * Ageo
        )

        self.Ageo = Ageo

    def calculate_Wfc(self):
        self.Wfc = 0.0

    # ============================================================
    # Secondary zone
    # ============================================================

    def calculate_secondary_zone_velocities(self):
        self.beta_2s = self.lib.radians(self.exit_blade_angle + self.delta_2s)

        test = self.calculate_W2s(self.W2p, n_iterations=100)
        if test != 0:
            self.calculate_W2s(test, n_iterations=500)

        self.Cm2s = self.W2s * self.lib.cos(self.beta_2s)
        self.Ctheta2s = self.exit_blade_speed + self.Cm2s * self.lib.tan(self.beta_2s)
        self.C2s = self.lib.sqrt(self.Cm2s**2 + self.Ctheta2s**2)

    def calculate_W2s(self, W2s_init, n_iterations=1000):
        hT2s = self.hT + self.Wfc
        W2s = W2s_init

        for _ in range(n_iterations):
            h2s = hT2s - 0.5 * W2s**2 + 0.5 * self.exit_blade_speed**2
            T2s = h2s / self.specific_heat
            rho2s = self.p2 / (self.gas_constant * T2s)

            Cm2s = (
                self.mass_flow / self.Ageo
                - self.rho2p * self.Cm2p * (1 - self.eps)
            ) / (rho2s * self.eps)

            W2s_new = Cm2s / self.lib.cos(self.beta_2s)

            if self.lib.abs(W2s_new - W2s) < 1e-5:
                self.W2s = W2s_new
                self.T2s = T2s
                self.rho2s = rho2s
                return 0

            W2s = W2s_new

        self.W2s = W2s_new
        self.T2s = T2s
        self.rho2s = rho2s
        return W2s

    # ============================================================
    # Averaging and optimization
    # ============================================================

    def calculate_mass_average_velocities(self):
        self.Cm2 = self.Cm2p * (1 - self.chi) + self.Cm2s * self.chi
        self.Ctheta2 = self.Ctheta2p * (1 - self.chi) + self.Ctheta2s * self.chi

    def make_2z_objective(self, Cm_ref, Ctheta_ref):
        self.Cm_ref = Cm_ref
        self.Ctheta_ref = Ctheta_ref

        def objective(x):
            self.chi, self.delta_2p, self.delta_2s = x
            self.calculate_primary_zone_velocities()
            self.calculate_epsilon()
            self.calculate_Wfc()
            self.calculate_secondary_zone_velocities()
            self.calculate_mass_average_velocities()

            return (
                (self.Cm2 - self.Cm_ref) ** 2
                + (self.Ctheta2 - self.Ctheta_ref) ** 2
            )

        return objective

    def constraint_equation(self, x):
        self.chi, self.delta_2p, self.delta_2s = x
        self.calculate_primary_zone_velocities()
        self.calculate_epsilon()
        self.calculate_Wfc()
        self.calculate_secondary_zone_velocities()
        self.calculate_mass_average_velocities()

        T02p = self.T2p + self.C2p / self.specific_heat
        T02s = self.T2s + self.C2s / self.specific_heat

        lhs = self.specific_heat * (
            T02p - self.T2p * (T02s / self.T2s)
        )
        rhs = 0.5 * (
            self.C2p**2 - (self.T2p / self.T2s) * self.C2s**2
        )

        return lhs - rhs
