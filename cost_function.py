import numpy as np
import cv2
import pickle as pkl
from  scipy.ndimage import gaussian_filter
import networkx as nx
import matplotlib.pyplot as plt


def add_line(img, arr, color=  (0,255,0), thick = 3):
    curr = img.copy()
    for i in range(len(arr) - 1):
        cv2.line(curr, (int(arr[i][0]), int(arr[i][1])), (int(arr[i + 1][0]), int(arr[i + 1][1])), color, thick)
    return curr

def cost_function(kart, start_point, end_point, path_points, track_graph, curr_track_img=None):
    total_cost = 0
    velocity = 0  # Starting velocity
    collision_point = False
    if curr_track_img is not None:
        curr_track_img = curr_track_img.copy()
    for i in range(1, len(path_points)):
        prev_point = path_points[i - 1]
        curr_point = path_points[i]
        
        # Calculate the distance between points
        distance = np.linalg.norm(np.array(curr_point) - np.array(prev_point))
        
        # Calculate angle difference for steering
        if i > 1:
            prev_vector = np.array(prev_point) - np.array(path_points[i - 2])
            curr_vector = np.array(curr_point) - np.array(prev_point)
            angle_diff = np.arctan2(np.cross(prev_vector, curr_vector), np.dot(prev_vector, curr_vector))
            # Penalize sharper turns more heavily
            turn_penalty = abs(angle_diff) * 10
        else:
            angle_diff = 0
            turn_penalty = 0
        
        # Update velocity based on acceleration and distance
        acceleration_time = min(distance / kart.acceleration, kart.max_speed)
        velocity = min(velocity + kart.acceleration * acceleration_time, kart.max_speed)
        
        # Penalty for exceeding maximum speed
        speed_penalty = max(0, velocity - kart.max_speed) * 10
        
        # Calculate total cost for this segment
        segment_cost = distance + speed_penalty + turn_penalty
        
        # Check for collisions
        grid_y, grid_x = curr_point[1] // 2, curr_point[0] // 2
        
        if not track_graph.has_node((grid_y, grid_x)):
            # Show the collision point on the graph
            collision_point = True
            if curr_track_img is not None:
                cv2.circle(curr_track_img, (int(curr_point[0]), int(curr_point[1])), 10, (255, 0, 0), -1)
            
            segment_cost += 100000  # High penalty for collision
        
        total_cost += segment_cost
    if curr_track_img is not None and collision_point:
        curr_track_img = add_line(curr_track_img, path_points)
        # for i in range(len(path_points)-1):
        #     cv2.line(curr_track_img, (int(path_points[i][0]), int(path_points[i][1])), (int(path_points[i+1][0]), int(path_points[i+1][1])), (0, 255, 0), 3)
        cv2.imshow('Track with collision points', curr_track_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return total_cost

def advanced_cost_function(kart, start_point, end_point, path_points, track_graph, curr_track_img=None, show = False):
    total_cost = 0
    velocity = 0  # Starting velocity
    collision_point = False
    fuel_consumption = 0
    if curr_track_img is not None:
        curr_track_img = curr_track_img.copy()
    
    for i in range(1, len(path_points)):
        prev_point = path_points[i - 1]
        curr_point = path_points[i]

        # Calculate the distance between points
        distance = np.linalg.norm(np.array(curr_point) - np.array(prev_point))

        # Calculate angle difference for steering
        if i > 1:
            prev_vector = np.array(prev_point) - np.array(path_points[i - 2])
            curr_vector = np.array(curr_point) - np.array(prev_point)
            angle_diff = abs(np.arctan2(np.cross(prev_vector, curr_vector), np.dot(prev_vector, curr_vector)))
            turn_penalty = angle_diff * 20  # Increased penalty for sharper turns

            # Adjust maximum speed based on the turning angle
            max_velocity_turn = kart.max_speed * (1 - angle_diff / np.pi)
        else:
            angle_diff = 0
            turn_penalty = 0
            max_velocity_turn = kart.max_speed

        # Update velocity based on acceleration/deceleration, distance, and turning radius
        if velocity < max_velocity_turn:
            velocity = min(velocity + kart.acceleration * (distance / kart.acceleration), max_velocity_turn)
        else:
            velocity = max(velocity - kart.braking * (distance / kart.braking), 0)

        # Fuel consumption based on acceleration/deceleration
        fuel_consumption += abs(kart.acceleration * (distance / kart.acceleration))

        # Speed Penalty
        speed_penalty = max(0, velocity - kart.max_speed) * 15

        # Calculate total cost for this segment
        segment_cost = distance + speed_penalty + turn_penalty + fuel_consumption

        # Check for collisions
        grid_y, grid_x = curr_point[1] // 2, curr_point[0] // 2

        if not track_graph.has_node((grid_y, grid_x)):
            # Show the collision point on the graph
            collision_point = True
            if curr_track_img is not None:
                cv2.circle(curr_track_img, (int(curr_point[0]), int(curr_point[1])), 10, (255, 0, 0), -1)

            segment_cost += 100000  # High penalty for collision

        total_cost += segment_cost

    if (curr_track_img is not None and collision_point) or show:
        curr_track_img = add_line(curr_track_img, path_points)
        cv2.imshow('Track with collision points', curr_track_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return total_cost


def interpolate_and_smooth(centerline):
    # Extract x and y coordinates of the centerline
    x = [point[0] for point in centerline]
    y = [point[1] for point in centerline]
    x = gaussian_filter(x, 3)
    y = gaussian_filter(y, 3)

    return np.array(zip(x, y))

