import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

def check_collision(point, track_graph):
    grid_y, grid_x = int(point[1] // 1), int(point[0] // 1)
    return track_graph.has_node((grid_y, grid_x))
        
def resample_path(centerline, track_width, num_points):
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

def smooth_path(centerline, sigma=2):
    x = [point[0] for point in centerline]
    y = [point[1] for point in centerline]
    x = gaussian_filter(x, sigma)
    y = gaussian_filter(y, sigma)
    return np.column_stack((x, y))



def mutate_path(path, track_boundaries, mutation_rate, mutation_strength, smoothing_radius):
    # Assume path is an array of points, track_boundaries provide feasible region information
    num_points = len(path)
    mutations = int(num_points * mutation_rate)
    
    for _ in range(mutations):
        # Select a mutation point
        idx = np.random.randint(0, num_points)
        
        # Determine mutation extent
        extent = int(mutation_strength * num_points)
        start = max(0, idx - extent)
        end = min(num_points, idx + extent + 1)
        
        # Generate a random lateral shift within track boundaries
        shift = np.random.uniform(-mutation_strength, mutation_strength)
        
        # Apply the mutation
        for i in range(start, end):
            if track_boundaries[path[i][0], path[i][1]]:  # Check if within boundaries
                path[i][1] += shift  # Apply lateral shift
        
        # Smooth the path to ensure transitions are not too abrupt
        path[start-smoothing_radius:end+1+smoothing_radius] = smooth_path(path[start-smoothing_radius:end+1+smoothing_radius], smoothing_radius)
    

def create_random_path(track_line, track_width, centerline, track_graph, weight=0.3):
    random_paths = []
    for i in range(len(track_line)):
        # Apply a random offset within the track width constraints
        random_offset = np.random.uniform(-track_width[i] * weight, track_width[i] * weight)
        random_point = track_line[i] + np.array([0, random_offset])

        # Ensure the random point does not go off the track
        if not check_collision(random_point,track_graph):
            # Adjust the random point to the nearest valid point if it's off the track
            random_point = track_graph.find_nearest_valid_point(random_point)

        random_paths.append(random_point)

    # Optionally, you can smooth the path here if needed
    random_paths = smooth_path(np.array(random_paths), sigma=2)
    return random_paths
def calculate_curvature(points):
    # Ensure points are in a numpy array for vectorized operations
    points = np.array(points)
    
    # First derivatives (Central differences)
    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])

    # Second derivatives (Central differences)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Curvature formula
    curvatures = np.abs(dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)

    return curvatures