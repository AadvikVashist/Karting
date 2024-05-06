import numpy as np
import cv2
import pickle as pkl
import networkx as nx
from scipy.spatial import distance
from heapq import heappop, heappush
from scipy.spatial import cKDTree
#import pchip interpolator
from scipy.interpolate import PchipInterpolator

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    contours = pkl.load(open('output.pkl', 'rb'))
    return img, contours

# Load the image and contours
img, top_contours = load_image('./output_img.jpg')


def extract_track_boundaries(image, contours):
    inner_contour = contours[1]
    outer_contour = contours[0]
    height, width = image.shape
    grid_size = 2

    grid_height = height // grid_size
    grid_width = width // grid_size
    grid = np.zeros((grid_height, grid_width), dtype=bool)

    for y in range(grid_height):
        for x in range(grid_width):
            cell_center = (x * grid_size + grid_size // 2, y * grid_size + grid_size // 2)
            if cv2.pointPolygonTest(outer_contour, cell_center, False) >= 0 and cv2.pointPolygonTest(inner_contour, cell_center, False) < 0:
                grid[y, x] = True

    graph = nx.grid_2d_graph(grid_height, grid_width)

    for y in range(grid_height):
        for x in range(grid_width):
            if not grid[y, x]:
                graph.remove_node((y, x))

    return graph, grid_size


graph, grid_size = extract_track_boundaries(img, top_contours)


import matplotlib.pyplot as plt
def find_centerline(img, inside_contour, outside_contour):
    centerline = []
    inside_contour = inside_contour.squeeze(axis=1)
    outside_contour = outside_contour.squeeze(axis=1)
    n = len(inside_contour)
    outside_contour_x = PchipInterpolator(np.arange(outside_contour.shape[0]), outside_contour[:, 1])
    outside_contour_y = PchipInterpolator(np.arange(outside_contour.shape[0]), outside_contour[:, 0])


    outside_contour = np.array([[outside_contour_y(i), outside_contour_x(i)] for i in np.arange(0,outside_contour.shape[0], 1)], dtype=int)
    new_img = np.zeros_like(img)
    cv2.drawContours(new_img, [inside_contour], -1, 255, 1)
    cv2.drawContours(new_img, [outside_contour], -1, 255, 1)
    cv2.imshow('Contours', new_img)
    cv2.waitKey(0)
    
    for i in range(n):
        # Get the previous, current, and next points for direction
        prev_point = inside_contour[i - 1 if i > 0 else n - 1]
        curr_point = inside_contour[i]
        next_point = inside_contour[(i + 1) % n]
        
        distances = np.linalg.norm(outside_contour - curr_point, axis=1)

        # Get indices of the 100 closest points
        closest_indices = np.argsort(distances)[:800]
        condensed_outside = outside_contour[closest_indices]

        filtered_outside = []
        curr_im = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for point in condensed_outside:
            # Convert point coordinates to grid coordinates
            weightage = 15
            y_dir= point[1]/weightage + curr_point[1]*(weightage-1)/weightage
            x_dir= point[0]/weightage + curr_point[0]*(weightage-1)/weightage
            grid_y = y_dir // grid_size
            grid_x = x_dir // grid_size

            
            # Check if the point belongs to the graph
            if graph.has_node((grid_y, grid_x)):
                filtered_outside.append(point)
                cv2.circle(curr_im, point , 3, (0, 255, 0), -1)
            else:
                x = 0
                cv2.circle(curr_im, point , 3, (0, 0, 255), -1)

        if filtered_outside and len(filtered_outside) != 1:
            projected_point = find_orthogonal_projection(curr_point, filtered_outside) 
        elif len(filtered_outside) == 1:
            projected_point = filtered_outside[0]
            # Find the midpoint between curr_point and projected_point
        else:
            print("No outside points found")
            continue
        midpoint = ((curr_point[0] + projected_point[0]) / 2, (curr_point[1] + projected_point[1]) / 2)
        # cv2.circle(curr_im, curr_point, 8, (255, 0, 0), -1) # the current point
        # cv2.circle(curr_im, [int(midpoint[0]), int(midpoint[1])], 8, (0, 100, 0), -1) # the midpoint
        # cv2.circle(curr_im, [int(projected_point[0]), int(projected_point[1])], 8, (0, 0, 100), -1) # the projected point
        # cv2.imshow('Select Start Point ' + str(i), curr_im)
        # cv2.waitKey(0)
        centerline.append(midpoint)

    return centerline
def find_orthogonal_projection(point, outside_contour):
    # get closest value in the outside contour to the point
    closest_point = min(outside_contour, key=lambda x: distance.euclidean(x, point))
    return closest_point



centerline = find_centerline(img, top_contours[1], top_contours[0])

# show the centerline
output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# for point in centerline:
#     cv2.circle(output_img, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)
#show the centerline as a line
pkl.dump(centerline, open('centerline.pkl', 'wb'))
pkl.dump(output_img, open('output_img.pkl', 'wb'))
pkl.dump(graph, open('graph.pkl', 'wb'))
for i in range(len(centerline)-1):
    cv2.line(output_img, (int(centerline[i][0]), int(centerline[i][1])), (int(centerline[i+1][0]), int(centerline[i+1][1])), (0, 255, 0), 1)
cv2.imshow('Centerline', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()