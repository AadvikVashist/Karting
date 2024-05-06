import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import pickle as pkl
from kart import Kart
from  cost_function import add_line, cost_function, advanced_cost_function
def modify_path(path):
    # Implement logic to modify the path slightly
    # For example, shift each point slightly based on random noise or a deterministic strategy
    new_path = path.copy()
    
    # Example: Move each point slightly randomly
    diff = np.random.randint(-20, 20, size=(len(new_path), 2))
    for i in range(1, len(new_path) - 1):
        new_path[i] = (new_path[i][0] + diff[i][0], new_path[i][1] + diff[i][1])
        # new_path[i] = (new_path[i][0] + np.random.randint(-50, 50), new_path[i][1] + np.random.randint(-50, 50))
    # import matplotlib.pyplot as plt
    # gauss_path = gaussian_filter(new_path, 3)
    new_path = np.array(new_path)
    x = new_path[:, 0]
    y= new_path[:, 1]
    gauss_path = np.column_stack((gaussian_filter(x, 3), gaussian_filter(y, 3)))
    # add the first point to the end
    gauss_path = np.vstack((gauss_path, gauss_path[0]))
    
    return gauss_path


def improve_line(kart, initial_path, track_graph, track_img, iterations=100, show = False):
    current_path = initial_path
    if show:
        show_image = track_img.copy()
    else:
        show_image = None
    current_cost = advanced_cost_function(kart, current_path[0], current_path[-1], current_path, track_graph, curr_track_img = show_image, show = show)
    for _ in range(iterations):
        print(f"Current cost: {current_cost} - Iteration: {_ + 1}")

        new_path = modify_path(current_path)

        # Evaluate the modified path
        new_cost = advanced_cost_function(kart, new_path[0], new_path[-1], new_path, track_graph, curr_track_img = show_image, show = show)
        
        # If the new path is better, update current path and cost
        if new_cost < current_cost:
            current_path = new_path
            current_cost = new_cost
    
    return current_path, current_cost


# Example kart with arbitrary values

track_img = cv2.imread('track_edited.jpg')
centerline = pkl.load(open('interpolated_centerline.pkl', 'rb'))

first_track_img = add_line(track_img, centerline)

# cv2.imshow('Select Start Point', first_track_img)
# cv2.setMouseCallback('Select Start Point', select_points)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Usage example:
# Assuming kart is an instance of the Kart class and path_points is a list of track points
optimal_path = centerline  # Calculate path_points (or obtain from centerline)
#add the first and last points
track_graph = pkl.load(open('graph.pkl', 'rb'))

kart = Kart(max_speed=30, acceleration=3, braking=5, turning_radius=1.5, width=1.0, length=1.5, steerability=1.0)

current_path, current_cost= improve_line(kart, optimal_path, track_graph,track_img, iterations=100000, show = False)

#display the improved path
output_img = cv2.imread('track_edited.jpg')
output_img = add_line(output_img, current_path)
cv2.imshow('Improved Path', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
pkl.dump(current_path, open('improved_path.pkl', 'wb'))

# cost = cost_function(kart, start_point, end_point, optimal_path, track_graph)
x=0
