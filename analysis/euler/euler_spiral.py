import numpy as np

def calculate_spiral(curvature, length):
    """Calculate the coordinates for a given curvature and length."""
    t = np.linspace(0, length, 100)
    s = np.sqrt(np.pi / curvature)
    x = np.cos(t**2 / (2 * s))
    y = np.sin(t**2 / (2 * s))
    return x, y

def euler_spiral_cornering(centerline, track_width_data, kart):
    optimized_line = [centerline[0]]
    for i in range(1, len(centerline)):
        prev_x, prev_y = optimized_line[-1]
        curvature = 1 / kart.turning_radius
        length = track_width_data[i] / 2
        
        x, y = calculate_spiral(curvature, length)

        adjusted_x = prev_x + x[-1]
        adjusted_y = prev_y + y[-1]

        if track_width_data[i] >= np.linalg.norm([adjusted_x, adjusted_y]):
            optimized_line.append((adjusted_x, adjusted_y))
        else:
            optimized_line.append(centerline[i])

    return optimized_line
