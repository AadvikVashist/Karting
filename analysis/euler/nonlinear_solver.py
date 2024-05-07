# nonlinear_solver.py
import scipy.optimize as opt
import numpy as np

def nonlinear_solver(centerline, track_width_data, kart):
    def objective(params):
        penalties = 0
        for i, p in enumerate(params):
            if p > kart.max_speed:
                penalties += (p - kart.max_speed) ** 2
        return np.sum(params**2) + penalties

    # Flatten centerline into a 1D array
    initial_guess = np.array(centerline).flatten()

    # Adjust bounds based on kart's max speed
    bounds = [(0, kart.max_speed)] * len(initial_guess)

    # Use a nonlinear solver to optimize
    result = opt.minimize(objective, initial_guess, bounds=bounds)
    
    # Reshape the result back to the original shape
    optimized_centerline = result.x.reshape((-1, 2))
    
    return optimized_centerline
