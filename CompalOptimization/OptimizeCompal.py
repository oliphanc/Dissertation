from pathlib import Path

from scipy.optimize import minimize, root
import numpy as np

import warnings
warnings.filterwarnings("ignore", module='scipy.optimize')


def get_2a(pump, primary_param, secondary_param, nstg=1):
    primary = pump.GetParameter(nstg, primary_param)
    secondary = pump.GetParameter(nstg, secondary_param)
    chi = pump.GetParameter(nstg, "MSEC_M")
    return primary*(1-chi)+secondary*chi

def get_Cm2a(pump, nstg=1):
    return get_2a(pump, "CM2P", "CM2S", nstg)
def get_Ct2a(pump, nstg=1):
    return get_2a(pump, "CT2P", "CT2S", nstg)
def get_H02a(pump, nstg=1):
    return get_2a(pump, "H02P", "H02S", nstg)
def get_T02a(pump, nstg=1):
    return get_2a(pump, "T02P", "T02S", nstg)

def get_parameters(pump, nstg=1):   
    p2a = pump.GetParameter(1,"P2")
    cm2a=get_Cm2a(pump)
    ct2a=get_Ct2a(pump)
    return np.array([p2a,cm2a,ct2a])

def find_v(x, cfd, pump, nstg):
    
    pump.SetParameter(1,"MR2",x[0])
    pump.SetParameter(1,"MSEC_M",x[1])
    pump.SetParameter(1,"DELTAPin",x[2])
    pump.SetParameter(1,"DELTAS",x[3])
    
    if pump.Run():
        results=get_parameters(pump,nstg)
        ret = cfd-results
    else:
        ret = np.array([1e12]*len(cfd))

    return ret

def d2s_constraint(ole_link, nstg=1):
    T2p = ole_link.GetParameter(1, "T2P")
    T2s = ole_link.GetParameter(1, "T2S")
    T02s = ole_link.GetParameter(1, "T02S")
    C2p = ole_link.GetParameter(1, "C2P")
    C2s = ole_link.GetParameter(1, "C2S")
    cp = ole_link.GetParameter(1, "Cp2M")
    C2si = np.sqrt(T2p/T2s) * C2s
    T02p = ole_link.GetParameter(1, "T02P")
    T02si = T2p * (T02s / T2s)
    H02si = T02si * cp
    H02p = T02p * cp
    con = H02si - H02p - 0.5 * (C2si**2 - C2p**2)
    return con

def calculate_mixed_outlet(df):
    P2 = ((df['P[Pa]']*df['dA[m^2]'])/df['dA[m^2]'].sum()).sum()
    Cm2 = ((df['Cm[m/s]']*df['Mass Flow[kg/s]'])/df['Mass Flow[kg/s]'].sum()).sum()
    Ct2 = ((df['Ct[m/s]']*df['Mass Flow[kg/s]'])/df['Mass Flow[kg/s]'].sum()).sum()

    return P2, Cm2, Ct2

def optimize_compal(compal_object, cfd,  x0=[0.2,-5, 0.]):

    mr2_sol = root(lambda x: find_v([*x, *x0], cfd, compal_object, 1)[0], [1.])

    bounds = ((0.01,.5), (-30, 0), (-90, 90))
    fun = lambda x: np.sum((find_v([*mr2_sol.x, *x], cfd, compal_object, 1)[1:])**2)
    def con(x):
        # design.SetParameter(1,"MR2",x[0])
        compal_object.SetParameter(1,"MSEC_M",x[0])
        compal_object.SetParameter(1,"DELTAPin",x[1])
        compal_object.SetParameter(1,"DELTAS",x[2])
        compal_object.Run()
        return d2s_constraint(compal_object, 1)

    constraints = ({'type':'eq', 'fun': con})
    sol = minimize(fun, x0, method='SLSQP', constraints=constraints, bounds=bounds)
    
    return mr2_sol, sol

if __name__ == "__main__":
    pass