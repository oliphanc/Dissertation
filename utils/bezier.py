import numpy as np
from scipy.special import comb # type: ignore
from scipy.optimize import minimize


def bernstein_poly(i, n, t):
    """
        The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * (t**i) * ((1 - t)**(n-i))


def bezier_curve(points, n=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        n is the number of points along the curve, defaults to 1000

    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, n)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def bezier_derivative(points, order: int = 1, nTimes: int = 1000):
    """
        Given a set of control points, return the 
        Bezier curve that is the exact derivate of the 
        curve defined by the control points.

        Points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        n is the number of points along the curve, defaults to 1000
    """
    points = np.asarray(points, dtype=float)
    m, d = points.shape  # m = degree + 1
    degree = m - 1

    if order < 0:
        raise ValueError("`order` must be non-negative.")
    if order > degree:
        raise ValueError(
            f"`order` ({order}) cannot exceed the curve degree ({degree})."
        )
    if order == 0:
        return bezier_curve(points, nTimes)

    # Compute k-th derivative control points
    ctrl = points.copy()
    coeff = 1.0
    for k in range(1, order + 1):
        coeff *= (degree - (k - 1))
        ctrl = coeff * (ctrl[1:] - ctrl[:-1])
    # 'ctrl' now has shape (degree-order+1, d)

    return bezier_curve(ctrl, nTimes)


def bezier_loss(interior_points, target_points, start, end):
    """
    Compute the loss function for fitting Bézier control points.
    'interior_points' contains only the control points between the fixed endpoints.
    """
    # Reshape interior points into (n,2) array and add fixed endpoints
    interior_points = interior_points.reshape(-1, 2)
    control_points = np.vstack([start, interior_points, end])
    
    # Generate the predicted bezier curve using the full set of control points
    predicted_points = bezier_curve(control_points)
    predicted_points = np.stack(predicted_points).T  # shape (n, 2)
    
    # Compute squared error loss
    loss = np.linalg.norm(predicted_points - target_points)**2
    return loss


def fit_bezier_curve(data_points, n_control_points):
    """
    Fit a Bézier curve to given data points by estimating control points.
    The endpoints of the curve are fixed to the first and last points in data_points.
    n_control_points is the total number of control points including endpoints.
    """
    # Fixed endpoints from the data points
    start = data_points[0]
    end = data_points[-1]
    
    # Number of interior points to optimize
    n_interior = n_control_points - 2
    if n_interior < 1:
        raise ValueError("There must be at least 1 interior control point.")
    
    # Initialize interior control points (linearly spaced between start and end)
    initial_interior = np.linspace(start, end, n_control_points)[1:-1]
    initial_guess = initial_interior.flatten()
    
    # Optimize only the interior control points
    result = minimize(
        bezier_loss,
        initial_guess,
        args=(data_points, start, end),
        method='BFGS'
    )
    optimized_interior = result.x.reshape(-1, 2)
    optimized_control_points = np.vstack([start, optimized_interior, end])
    return optimized_control_points, result.fun

def fit_constant_x_bezier_curve(data_points, n_control_points):
    """
        Fit a Bézier curve to given data points by estimating control points. Ensures that the 'x' values of the
        control points maintain constant spacing
    """
    def loss(yPoints, xPoints, endpoints, yTrue):
        # Insert the fixed endpoints in correct order: start then end
        yPoints = np.insert(yPoints, 0, endpoints[0])
        yPoints = np.append(yPoints, endpoints[1])
        points = np.stack([xPoints, yPoints]).T
        _, yNew = bezier_curve(points)
        return np.linalg.norm((yTrue - yNew)**2)
    
    xmin = data_points[:, 0].min()
    xmax = data_points[:, 0].max()
    # Use n_control_points for equally spaced x values
    xPoints = np.linspace(xmin, xmax, n_control_points)
    # Initialize interior control points with the mean y value (n_control_points - 2 because endpoints are fixed)
    initial_guess = np.array([data_points[:, 1].mean()] * (n_control_points - 2))
    # Endpoints: first y value, last y value
    endpoints = [data_points[0, 1], data_points[-1, 1]]
    
    # Optimize control points to minimize the loss
    result = minimize(
        loss,
        initial_guess,
        args=(xPoints, endpoints, data_points[:, 1]),
        method='BFGS'
    )
    
    # Reassemble the optimized control points with fixed endpoints (correct order)
    yPoints = result.x
    yPoints = np.insert(yPoints, 0, endpoints[0])
    yPoints = np.append(yPoints, endpoints[1])
    optimized_control_points = np.stack([xPoints, yPoints]).T

    return optimized_control_points, result.fun