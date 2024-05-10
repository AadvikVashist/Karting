import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import cv2
import pickle as pkl


def check_collision(point, track_graph):
    grid_y, grid_x = int(point[1]), int(point[0])
    return not track_graph[grid_y, grid_x]
        
    
def mutate(centerline, path, mutation_rate, mutation_magnitude, track_width, track_graph, smoothen):
    connection_buffer = int(np.clip(int(len(path) * 0.1), 1, 25))
    i = 0
    
    path = np.array(path) 
    mutation_min = np.clip(len(path) * 0.1 * mutation_rate, 5, 50)
    mutation_max = np.clip(len(path) * mutation_rate, 1, 500)
    path_appended = np.vstack((path[-connection_buffer::], path, path[0:connection_buffer])) #append the first x points to the end of the array
    #do the same for the track_width and centerline
    # centerline_appended = np.vstack((centerline[-connection_buffer::], centerline, centerline[0:connection_buffer]))
    # width_appended = np.hstack((track_width[-connection_buffer::], track_width, track_width[0:connection_buffer]))
    
    track_width = np.array(track_width)

    
    while i < len(path_appended) - 1:
        
        if np.random.rand() < mutation_rate:
            random_ret = np.random.uniform(-mutation_magnitude, mutation_magnitude)
            quant = int(np.round(np.random.uniform(low=mutation_min, high=mutation_max)))

            if i + quant >= len(path_appended):
                quant = len(path_appended) - i - 1
            if quant >= 3:
                x = np.linspace(-1, 1, quant)
                gaussian_weights = np.exp(-0.5 * (x ** 2) / (0.3 ** 2)) * random_ret
            else:
                gaussian_weights = [random_ret] * quant
            
            updated_points = update_consecutive_points(path_appended[i:i + quant], gaussian_weights, clockwise=True)
            path_appended[i:i + quant] = updated_points
            path_appended[i-5:i+quant+5] = smooth_centerline(path_appended[i-5:i+quant+5], 1)

            i += quant
        i += 1

    # path_appended = smooth_centerline(path_appended, sigma)
    #take the average between the appended 5 points and the original points
    if smoothen:
        sigma = np.around(np.clip(np.random.uniform(-100, 1), 0,1))
        if sigma > 0:
            path_appended = smooth_centerline(path_appended, sigma)
        
    connect_ends = np.mean([path_appended[0:connection_buffer*2], path_appended[len(path)::]],axis=0)
    ret_path = np.vstack((connect_ends[-connection_buffer:], path_appended[2*connection_buffer:-connection_buffer*2], connect_ends[0:connection_buffer] ))
    
    
    #collision check
    
    check = [check_collision(point, track_graph) for point in centerline]

    if any(check) > 0:
        #get the indices of the issues
        indices = [i for i, x in enumerate(check) if x > 0]
        #get the points that are causing the issues
        for index in indices:
            new_pos = ret_path[index]
            # Find the nearest centerline point within 2 * track_width
            near_centerline_points = centerline[np.linalg.norm(centerline - new_pos, axis=1) < 2 * track_width]
            if len(near_centerline_points) > 0:
                closest_point = near_centerline_points[np.argmin(np.linalg.norm(near_centerline_points - new_pos, axis=1))]
            while check_collision(new_pos, track_graph) > 0:
                new_pos = (new_pos * 7 + closest_point) / 8
            ret_path[index] = new_pos

    return ret_path


def update_consecutive_points(path, weights, clockwise=True):
    # This function assumes `path` is an array of numpy arrays [np.array([x, y]), ...]
    updated_path = np.copy(path)
    n = len(path)

    for i in range(n):
        # Retrieve the current point and its neighbors
        current_point = path[i]
        prev_point = path[i - 1 if i > 0 else n - 1]
        next_point = path[(i + 1) % n]

        # Calculate the normal direction
        # Vector from previous to current point
        v1 = current_point - prev_point
        # Vector from current to next point
        v2 = next_point - current_point
        # Average vector - this approximates the tangent direction at the current point
        tangent = (v1 + v2) / 2
        
        if np.linalg.norm(tangent) == 0:
            # Tangent is a zero vector; use just one of the vectors for the tangent
            tangent = v1 if np.linalg.norm(v1) > 0 else v2
        
        # Calculate the normal: rotate the tangent by 90 degrees
        if np.linalg.norm(tangent) > 0:
            if clockwise:
                normal = np.array([-tangent[1], tangent[0]])  # Rotate clockwise
            else:
                normal = np.array([tangent[1], -tangent[0]])  # Rotate counterclockwise
            
            normal /= np.linalg.norm(normal)  # Normalize the normal vector

            # Apply the weight to the normal to calculate the shift
            shift = normal * weights[i]
            updated_path[i] = current_point + shift
        else:
            # If tangent and thus normal cannot be determined, do not change the point
            updated_path[i] = current_point

    return updated_path

def resample_centerline(centerline, track_width, num_points):
    # Calculate cumulative distances
    diffs = np.diff(centerline, axis=0)
    segment_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
    
    # Generate new sampling points
    new_distances = np.linspace(0, cumulative_lengths[-1], num_points)
    
    # Interpolate centerline points
    x_interp = interp1d(cumulative_lengths, centerline[:, 0], kind='linear')
    y_interp = interp1d(cumulative_lengths, centerline[:, 1], kind='linear')
    new_centerline = np.vstack([x_interp(new_distances), y_interp(new_distances)]).T
    
    # Interpolate track width
    width_interp = interp1d(cumulative_lengths, track_width, kind='linear')
    new_track_width = width_interp(new_distances)
    
    return new_centerline, new_track_width

def smooth_centerline(centerline, sigma=2):
    x = [point[0] for point in centerline]
    y = [point[1] for point in centerline]
    x = gaussian_filter(x, sigma)
    y = gaussian_filter(y, sigma)
    return np.column_stack((x, y))



def update_point(point, point_b4, point_aft, weight=0.3, clockwise=True):
    # Update a point based on the points before and after it
    direction = point_aft - point_b4
    direction = [direction[1], -direction[0]]
    # check ortho using dot product
    x = np.dot(direction, point_aft - point_b4)
    if abs(x) > 1e-6:
        raise ValueError("The points are not orthogonal")
    direction = direction / np.linalg.norm(direction)
    #ensure that it is clockwise relative to the centerline
    if clockwise and np.dot(direction, point - point_b4) < 0:
            direction = -direction
    
    new_point = point + direction * weight
    # plt.plot(new_point[0], new_point[1], 'ro', label='Random Path')
    # plt.plot(point[0], point[1], 'bo', label='Centerline')
    # plt.plot(point_b4[0], point_b4[1], 'go', label='previous Path')
    # plt.plot(point_aft[0], point_aft[1], 'o', label='after Path')

    # plt.plot([point[0], new_point[0]], [point[1], new_point[1]])
    # plt.plot([point_b4[0], point_aft[0]], [point_b4[1], point_aft[1]])
    # plt.legend()
    # plt.show()
    return new_point

def create_random_path(track_line, track_width, centerline, track_graph, weight = 0.3):
    # Create a random path within the track boundaries
    random_paths = []
    #add 0 index to end of centerline array
    connection_buffer = np.clip(int(len(track_line) * 0.1), 1, 25)
    for i in range(0, len(track_line)):
        if i >= connection_buffer and i < len(track_line) - connection_buffer:
            random_offset = np.random.uniform(-track_width[i] * weight, track_width[i] * weight)
            # get the point before, and after the current point in the centerline to get the direction
            # get the vector between the two points
            random_path_point = update_point(track_line[i], track_line[i - 1], track_line[i + 1], random_offset)
            
            # if check_collision(random_path_point, track_graph) > 0:
            #     # Find the nearest centerline point within 2 * track_width
            #     near_centerline_points = centerline[np.linalg.norm(centerline - random_path_point, axis=1) < 2 * track_width]
            #     if len(near_centerline_points) > 0:
            #         closest_point = near_centerline_points[np.argmin(np.linalg.norm(near_centerline_points - random_path_point, axis=1))]

            #     while check_collision(random_path_point, track_graph) > 0:
            #         plt.plot(random_path_point[0], random_path_point[1], 'ro', label='Random Path')
            #         plt.plot(closest_point[0], closest_point[1], 'go', label='Closest Point')
            #         plt.plot(centerline[:, 0], centerline[:, 1], 'bo', label='Centerline')
            #         plt.plot(track_line[:, 0], track_line[:, 1], 'yo', label='Track Line')
            #         plt.legend()
            #         plt.show()
            #         random_path_point = (random_path_point * 2 + closest_point) / 3
            random_paths.append(random_path_point)

        else:
            random_paths.append(track_line[i])    
    # cv2.imshow("Random Path", add_line(track_img.copy(), random_paths))
    # cv2.waitKey(1)
    return np.array(random_paths)
