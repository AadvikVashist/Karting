import numpy as np
import cv2
import pickle as pkl
from  scipy.ndimage import gaussian_filter
import networkx as nx
import matplotlib.pyplot as plt
import time

def add_line(img, arr, color=  (0,255,0), thick = 3):
    curr = img.copy()
    for i in range(len(arr) - 1):
        cv2.line(curr, (int(arr[i][0]), int(arr[i][1])), (int(arr[i + 1][0]), int(arr[i + 1][1])), color, thick)
    return curr


#chatgpt added comments to this
def optimized_cost_function_1(kart, path_points, track_graph, curr_track_img=None, show=False, use_kart_width=False):
    total_cost = 0  # Initialize the total cost of the path
    velocity = 0  # Starting velocity of the kart
    collision_point = False  # Flag to indicate if there was a collision

    # Precompute constants
    max_speed = kart.max_speed  # Maximum speed of the kart
    acceleration = kart.acceleration  # Acceleration rate of the kart
    braking = kart.braking  # Braking rate of the kart
    max_speed_penalty = 15 * max(0, max_speed - kart.max_speed)  # Penalty for exceeding max speed
    kart_half_width = kart.width / 2  # Half of the kart's width for collision detection

    if curr_track_img is not None:
        curr_track_img = curr_track_img.copy()  # Create a copy of the current track image for visualization

    path_points = np.array(path_points)  # Convert path points to a NumPy array for easier manipulation
    path_points = np.append(path_points, path_points[0:100], axis=0)  # Make the path continuous for visualization
    distances = np.linalg.norm(np.diff(path_points, axis=0), axis=1)  # Calculate distances between consecutive points

    for i in range(1, len(path_points)):
        distance = distances[i - 1]  # Get the distance between current and previous path point

        if i > 1:
            # Calculate angle difference for steering penalties
            prev_vector = path_points[i - 2] - path_points[i - 1]  # Vector from two steps back to the previous point
            curr_vector = path_points[i - 1] - path_points[i]  # Vector from the previous point to the current point
            angle_diff = np.abs(np.arctan2(np.cross(prev_vector, curr_vector), np.dot(prev_vector, curr_vector)))
            turn_penalty = angle_diff * 20  # Penalty for sharper turns based on angle difference
            max_velocity_turn = max_speed * (1 - angle_diff / np.pi)  # Max velocity allowed for the current turn
        else:
            angle_diff = 0  # No angle difference for the first point
            turn_penalty = 0  # No turn penalty for the first point
            max_velocity_turn = max_speed  # Max velocity for the first point
            curr_vector = path_points[i - 1] - path_points[i]  # Vector from the previous point to the current point

        # Update velocity considering acceleration, braking, and turn constraints
        velocity = np.clip(velocity + acceleration * (distance / acceleration) * (velocity < max_velocity_turn) -
                           braking * (distance / braking) * (velocity > max_velocity_turn), 0, max_velocity_turn)

        speed_penalty = max_speed_penalty * np.maximum(0, velocity - max_speed)  # Penalty for exceeding speed limit

        segment_cost = distance + speed_penalty + turn_penalty  # Calculate total cost for this segment
        if use_kart_width:
            kart_direction = curr_vector / np.linalg.norm(curr_vector)  # Normalize current vector to get direction
            kart_perpendicular = np.array([-kart_direction[1], kart_direction[0]])  # Get perpendicular direction
            kart_left = path_points[i] + kart_half_width * kart_perpendicular  # Calculate kart's left edge
            kart_right = path_points[i] - kart_half_width * kart_perpendicular  # Calculate kart's right edge

            for point in [path_points[i], kart_left, kart_right]:  # Check central, left, and right points for collision
                grid_y, grid_x = point[1] // 2, point[0] // 2  # Convert to grid coordinates

                if not track_graph.has_node((grid_y, grid_x)):  # Check if this point is within the track
                    collision_point = True  # Mark that a collision occurred
                    if curr_track_img is not None:
                        cv2.circle(curr_track_img, (int(point[0]), int(point[1])), 10, (255, 0, 0), -1)  # Mark collision
                    segment_cost += 100 * 1000  # Add a high penalty for collision
                    break  # Exit loop after detecting a collision
        else:
            # Check for collisions
            grid_y, grid_x = path_points[i, 1] // 2, path_points[i, 0] // 2

            if not track_graph.has_node((grid_y, grid_x)):
                # Show the collision point on the graph
                collision_point = True
                if curr_track_img is not None:
                    cv2.circle(curr_track_img, (int(path_points[i, 0]), int(path_points[i, 1])), 10, (255, 0, 0), -1)

                segment_cost += 100*1000  # High penalty for collision

        total_cost += segment_cost  # Add segment cost to the total cost

    if (curr_track_img is not None and collision_point) or show:
        curr_track_img = add_line(curr_track_img, path_points)  # Draw the path line on the track image
        cv2.imshow('Track with collision points', curr_track_img)  # Show the track with collision points
        cv2.waitKey(0)  # Wait for a key press
        cv2.destroyAllWindows()  # Close the window

    return total_cost  # Return the total cost


def optimized_cost_function(kart, path_points, track_graph, curr_track_img=None, show=False, use_kart_width=False):
    total_cost = 0
    velocity = 0
    collision_point = False

    max_speed = kart.max_speed
    acceleration = kart.acceleration
    braking = kart.braking
    max_speed_penalty = 10 * max(0, max_speed - kart.max_speed)
    kart_half_width = kart.width / 2

    if curr_track_img is not None:
        curr_track_img = curr_track_img.copy()

    path_points = np.array(path_points)
    distances = np.linalg.norm(np.diff(path_points, axis=0), axis=1)

    for i in range(1, len(path_points)):
        distance = distances[i - 1]

        if i > 1:
            prev_vector = path_points[i - 2] - path_points[i - 1]
            curr_vector = path_points[i - 1] - path_points[i]
            angle_diff = np.abs(np.arctan2(np.cross(prev_vector, curr_vector), np.dot(prev_vector, curr_vector)))

            # Apply an exponential penalty to sharp turns to further optimize straights
            turn_penalty = (angle_diff ** 2) * 100
            max_velocity_turn = max_speed * np.clip(1 - angle_diff / np.pi, 0.2, 1)
        else:
            angle_diff = 0
            turn_penalty = 0
            max_velocity_turn = max_speed
            curr_vector = path_points[i - 1] - path_points[i]

        velocity = np.clip(velocity + acceleration * distance / acceleration * (velocity < max_velocity_turn) -
                           braking * distance / braking * (velocity > max_velocity_turn), 0, max_velocity_turn)

        speed_penalty = max_speed_penalty * np.maximum(0, velocity - max_speed)

        segment_cost = distance + speed_penalty + turn_penalty

        if use_kart_width:
            kart_direction = curr_vector / np.linalg.norm(curr_vector)
            kart_perpendicular = np.array([-kart_direction[1], kart_direction[0]])
            kart_left = path_points[i] + kart_half_width * kart_perpendicular
            kart_right = path_points[i] - kart_half_width * kart_perpendicular

            for point in [path_points[i], kart_left, kart_right]:
                grid_y, grid_x = point[1] // 2, point[0] // 2

                if not track_graph.has_node((grid_y, grid_x)):
                    collision_point = True
                    if curr_track_img is not None:
                        cv2.circle(curr_track_img, (int(point[0]), int(point[1])), 10, (255, 0, 0), -1)
                    segment_cost += 100 * 1000
                    break
        else:
            grid_y, grid_x = path_points[i, 1] // 2, path_points[i, 0] // 2

            if not track_graph.has_node((grid_y, grid_x)):
                collision_point = True
                if curr_track_img is not None:
                    cv2.circle(curr_track_img, (int(path_points[i, 0]), int(path_points[i, 1])), 10, (255, 0, 0), -1)

                segment_cost += 100 * 1000

        total_cost += segment_cost

    if (curr_track_img is not None and collision_point) or show:
        curr_track_img = add_line(curr_track_img, path_points)
        cv2.imshow('Track with collision points', curr_track_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return total_cost
