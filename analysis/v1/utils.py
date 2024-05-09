import os
import numpy as np
cwd = os.getcwd()


def get_base_file_path(file):
    return os.path.join(cwd, file)
def line_length(points):
    # Convert the list of points to a NumPy array if it's not already one
    points = np.array(points)
    
    # Calculate the differences between consecutive points
    diff = np.diff(points, axis=0)
    
    # Calculate the Euclidean distance between consecutive points
    distances = np.sqrt(np.sum(diff**2, axis=1))
    
    # Return the sum of the distances
    return np.sum(distances)