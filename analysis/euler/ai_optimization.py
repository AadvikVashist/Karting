# ai_optimization.py
import random
import numpy as np
def ai_optimization(centerline, track_width_data, kart):
    optimized_line = np.array(centerline)  # Ensure it's a NumPy array

    for _ in range(100):  # Example iteration count
        index = random.randint(0, len(centerline) - 1)
        delta = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])  # Adjust both x and y

        optimized_line[index] += delta

        # Enforce kart-specific constraints and consider track width
        track_width = track_width_data[index]
        optimized_line[index][0] = max(0, min(optimized_line[index][0], kart.max_speed))
        optimized_line[index][1] = max(0, min(optimized_line[index][1], kart.max_speed))

    return optimized_line.tolist()
