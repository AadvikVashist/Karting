import numpy as np
import cv2
import pickle as pkl
from  scipy.ndimage import gaussian_filter
import networkx as nx
import matplotlib.pyplot as plt
import time
from utils import line_length
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def plot_path_with_costs(track_graph, path_points, point_costs, collision_points):
    fig, ax = plt.subplots()
    if track_graph is not None:
        ax.imshow(track_graph, extent=(0, 10, 0, 10))  # Adjust extents as necessary
    
    path_points = np.array(path_points)
    point_costs = np.array(point_costs)
    norm = Normalize(vmin=min(point_costs), vmax=max(point_costs))
    cmap = cm.get_cmap('hot')  # Choose a colormap that suits your preferences
    
    scatter = ax.scatter(path_points[:, 0], path_points[:, 1], c=point_costs, cmap=cmap, norm=norm)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Cost')

    for point in collision_points:
        ax.plot(point[0], point[1], 'rx')  # Mark collision points with red crosses

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Path Cost Heatmap')
    plt.show()
    
def add_line(img, arr, color=  (0,255,0), thick = 3):
    curr = img.copy()
    for i in range(len(arr) - 1):
        cv2.line(curr, (int(arr[i][0]), int(arr[i][1])), (int(arr[i + 1][0]), int(arr[i + 1][1])), color, thick)
    return curr


def optimized_cost_function(kart, path_points, track_graph, curr_track_img=None, show=False, min_distance=0):
    total_cost = 0
    velocity = kart.max_speed  # Initial velocity
    collision_points = []
    point_cost = [0]
    for i in range(1, len(path_points)):
        if i == 1:
            prev_vector = path_points[i - 1]  # Special handling for the first point
        else:
            prev_vector = path_points[i - 1] - path_points[i - 2]
        curr_vector = path_points[i] - path_points[i - 1]
        angle_diff = np.arctan2(np.linalg.norm(np.cross(prev_vector, curr_vector)), np.dot(prev_vector, curr_vector))
        
        # Increased penalty for sharp turns
        turn_penalty = np.exp(-np.cos(angle_diff)) * 150  # Adjust the multiplier to increase the penalty
        
        # Velocity adjusted for turn sharpness
        max_velocity_turn = kart.max_speed * np.clip(1 - angle_diff / np.pi, 0.2, 1)
        
        # Acceleration penalty in turns
        acceleration_penalty = 0
        if angle_diff > np.pi / 4:  # Assuming π/4 as a threshold for "sharp" turns
            acceleration_penalty = 50 * np.clip(kart.acceleration * (1 - np.cos(angle_diff)), 0, kart.acceleration)

        # Very sharp turn penalty
        extra_sharp_turn_penalty = 0
        if angle_diff > np.pi / 3:  # More severe penalty for very sharp turns
            extra_sharp_turn_penalty = 200
        
        # Dynamic velocity adjustment
        velocity_change = np.clip(max_velocity_turn - velocity, -kart.braking, kart.acceleration)
        velocity = np.clip(velocity + velocity_change, 0, max_velocity_turn)
        
        distance = np.linalg.norm(curr_vector)
        speed_penalty = 10 * max(0, velocity - kart.max_speed)
        segment_cost = distance + speed_penalty + turn_penalty + acceleration_penalty + extra_sharp_turn_penalty
        
        # Collision detection and adding collision cost
        collision_cost = check_collision(path_points[i], kart, track_graph, curr_track_img)
        if collision_cost > 0:
            collision_points.append(path_points[i])
        point_cost.append(segment_cost )
        segment_cost += collision_cost
        total_cost += segment_cost
    
    # Optional visualization of track and collisions
    if show and curr_track_img is not None and collision_points:
        # display_track(curr_track_img, path_points, collision_points)
        plot_path_with_costs(curr_track_img, path_points, point_cost, collision_points)
    return total_cost, collision_points

def check_collision(point, kart, track_graph, curr_track_img = None):
    collision_cost = 0

    grid_y, grid_x = int(point[1] // 2), int(point[0] // 2)
    if not track_graph.has_node((grid_y, grid_x)):
        collision_cost += 1000*1000
        if curr_track_img is not None:
            cv2.circle(curr_track_img, (int(point[0]), int(point[1])), 10, (255, 0, 0), -1)

    return collision_cost

def display_track(img, path_points):
    img = add_line(img, path_points)  # Assumes implementation of add_line function
    cv2.imshow('Track with collision points', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()