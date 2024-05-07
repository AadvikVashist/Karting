# integrated_approach.py
from euler_spiral import euler_spiral_cornering
from nonlinear_solver import nonlinear_solver
from ai_optimization import ai_optimization
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

def visualize_line(lines, contours, title):
    plt.figure()
    for name, line in lines.items():
        x, y = zip(*line)
        y = np.array(y)
        plt.plot(x, -1*y, label=name)

    plt.plot(contours[0][:,0, 0], -1*contours[0][:, 0, 1], 'k-', label='Inner Boundary')
    plt.plot(contours[1][:,0, 0], -1*contours[1][:, 0, 1], 'k-', label='Outer Boundary')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()

def integrated_approach(centerline, track_width_data, contours, kart):
    # Apply Euler Spiral cornering
    lines = {}
    lines["Centerline"] = centerline
    print("Applying Euler Spiral cornering...")
    line = euler_spiral_cornering(centerline, track_width_data, kart)
    lines["Euler Spiral"] = line
    visualize_line(lines, contours, "After Euler Spiral Cornering")
    
    # Apply nonlinear solver
    print("Applying Nonlinear Solver...")
    line = nonlinear_solver(line, track_width_data, kart)
    lines["Nonlinear Solver"] = line
    # visualize_line(lines, contours, "After Nonlinear Solver")
    
    # Apply AI optimization
    print("Applying AI Optimization...")
    optimized_line = ai_optimization(line, track_width_data, kart)
    lines["AI Optimization"] = optimized_line
    visualize_line(lines, contours, "After AI Optimization")
    
    return optimized_line