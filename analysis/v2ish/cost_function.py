import numpy as np
import matplotlib.pyplot as plt
from path_manipulation import check_collision

# from collections import OrderedDict
# cost_function_cache = OrderedDict()
# CACHE_SIZE_LIMIT = 1000


from collections import OrderedDict
import hashlib
class CostFunctionCache:
    def __init__(self, cache_size_limit=1000):
        self.cache = OrderedDict()
        self.cache_size_limit = cache_size_limit

    def _hash_path(self, path):
        # Use a hash of the path to reduce memory usage
        return hashlib.md5(np.array(path).tobytes()).hexdigest()

    def get(self, kart, path, track_graph):
        key = (self._hash_path(path), hash(track_graph))
        if key in self.cache:
            return self.cache.pop(key)
        else:
            cost, collision_points = calculate_cost(kart, path, track_graph)
            self.set(key, (cost, collision_points))
            return cost, collision_points

    def set(self, key, value):
        if len(self.cache) >= self.cache_size_limit:
            self.cache.popitem(last=False)  # Pop the oldest item
        self.cache[key] = value
        self.cache.move_to_end(key, last=True)

# def memoized_optimized_cost_function(kart, path, track_graph, min_dist, curr_track_img = None, show = False):
#     # Use tuple of parameters as key to check the cache
#     cache_key = (tuple(map(tuple, path)), tuple(track_graph))
    
#     if cache_key in cost_function_cache:
#         return cost_function_cache.pop(cache_key)
    
#     # Compute the result if not in cache
#     cost,collision_points = optimized_cost_function(kart, path, track_graph, min_distance=min_dist, curr_track_img=curr_track_img, show=show)
#     cost_function_cache[cache_key] = [cost,collision_points]
    
#     # Move the recently used item to the end of the cache
#     cost_function_cache.move_to_end(cache_key)

#     # Clear cache entries if the size exceeds the limit
#     if len(cost_function_cache) > CACHE_SIZE_LIMIT:
#         cost_function_cache.popitem(last=False)
    
#     return cost, collision_points


def visualize_cost_factors(path, total_time, collision_penalty, smoothness_penalty, speed_penalty):
    plt.figure(figsize=(10, 5))
    labels = ['Total Time', 'Collision Penalty', 'Smoothness Penalty', 'Speed Penalty']
    values = [total_time, collision_penalty, smoothness_penalty, speed_penalty]
    colors = ['blue', 'red', 'green', 'purple']
    plt.bar(labels, values, color=colors)
    plt.title('Visualization of Cost Components')
    plt.ylabel('Cost')
    plt.show()
    
    
def calculate_cost(path, kart, track_graph, visualization=False):
    total_time = 0
    collision_penalty = 0
    smoothness_penalty = 0
    speed_penalty = 0
    
    # Constants for penalties
    COLLISION_COST = 1000
    TURN_COST_MULTIPLIER = 5
    SPEED_COST_MULTIPLIER = 2

    # Iterate through the path
    for i in range(1, len(path)):
        current_point = path[i]
        previous_point = path[i-1]

        # Time calculation
        distance = np.linalg.norm(current_point - previous_point)
        speed = min(kart.max_speed, kart.get_max_speed(kart.get_curvature(path[i-5:i+5])))
        time_increment = distance / speed
        total_time += time_increment

        # Collision check
        if not check_collision(current_point, track_graph):
            collision_penalty += COLLISION_COST

        # Smoothness calculation
        if i > 1:
            prev_vector = previous_point - path[i-2]
            curr_vector = current_point - previous_point
            angle_diff = np.arccos(np.clip(np.dot(prev_vector, curr_vector) / 
                            (np.linalg.norm(prev_vector) * np.linalg.norm(curr_vector)), -1.0, 1.0))
            smoothness_penalty += TURN_COST_MULTIPLIER * angle_diff

        # Speed consistency
        if i > 1:
            previous_speed = np.linalg.norm(previous_point - path[i-2]) / time_increment
            speed_diff = abs(speed - previous_speed)
            speed_penalty += SPEED_COST_MULTIPLIER * speed_diff

    # Total cost calculation
    total_cost = total_time + collision_penalty + smoothness_penalty + speed_penalty

    # Visualization of the cost factors
    if visualization:
        visualize_cost_factors(path, total_time, collision_penalty, smoothness_penalty, speed_penalty)

    return total_cost



def calculate_vectorized_cost(path, kart, track_graph, cache, visualization=False):
    distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
    speeds = np.array([cache.get_speed_at_point(point, kart) for point in path[:-1]])
    times = distances / speeds
    total_time = np.sum(times)

    collisions = np.array([cache.check_collision(point, track_graph) for point in path])
    collision_penalty = np.sum(collisions) * 1000  # Assuming a flat penalty for collisions

    angles = np.arccos(np.clip(np.einsum('ij,ij->i', np.diff(path[:-1], axis=0), np.diff(path[1:], axis=0)) /
                        (np.linalg.norm(np.diff(path[:-1], axis=0), axis=1) * np.linalg.norm(np.diff(path[1:], axis=0), axis=1)), -1.0, 1.0))
    smoothness_penalty = np.sum(angles) * 5  # Adjust the smoothness penalty multiplier

    # Speed consistency
    speed_diffs = np.abs(np.diff(speeds))
    speed_penalty = np.sum(speed_diffs) * 2  # Adjust the speed penalty multiplier

    total_cost = total_time + collision_penalty + smoothness_penalty + speed_penalty

    # Visualization of the cost factors
    if visualization:
        visualize_cost_factors(path, total_time, collision_penalty, smoothness_penalty, speed_penalty)

    return total_cost