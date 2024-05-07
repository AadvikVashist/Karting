import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter
from .utils import get_base_file_path
#gaussian blur 1d

def load_centerline(filename):
    with open(filename, 'rb') as f:
        centerline = pickle.load(f)
    return centerline


def interpolate_and_smooth(centerline):
    # Extract x and y coordinates of the centerline
    x = [point[0] for point in centerline]
    y = [point[1] for point in centerline]
    x = gaussian_filter(x, 3)
    y = gaussian_filter(y, 3)
    # Create PchipInterpolator objects for x and y coordinates
    interpolator_x = PchipInterpolator(np.arange(len(x)), x)
    interpolator_y = PchipInterpolator(np.arange(len(y)), y)

    # Generate new indices for interpolation
    new_indices = np.linspace(0, len(x) - 1, num=1000)

    # Interpolate x and y coordinates using the new indices
    interpolated_x = interpolator_x(new_indices)
    interpolated_y = interpolator_y(new_indices)

    # Combine interpolated x and y coordinates into a new centerline
    interpolated_centerline = np.column_stack((interpolated_x, interpolated_y))

    return interpolated_centerline

def visualize_differences(original_centerline, interpolated_centerline, img_path):
    # Load the image
    img = cv2.imread(img_path)

    # Plot original centerline
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    original_centerline = np.array(original_centerline)
    
    plt.plot(original_centerline[:, 0], original_centerline[:, 1], color='red', label='Original Centerline')
    plt.title('Original Centerline')
    plt.axis('off')

    # Plot interpolated centerline
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.plot(interpolated_centerline[:, 0], interpolated_centerline[:, 1], color='blue', label='Interpolated Centerline')
    plt.title('Interpolated Centerline')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Plot differences in 2D plots
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.linspace(0,1000, original_centerline.shape[0]), original_centerline[:, 0], color='red', label='Original Centerline')
    plt.plot(np.linspace(0,1000, interpolated_centerline.shape[0]), interpolated_centerline[:, 0], color='blue', label='Interpolated Centerline')
    plt.title('Original vs Interpolated Centerline')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.linspace(0,1000, original_centerline.shape[0]), original_centerline[:, 1], color='red', label='Original Centerline')
    plt.plot(np.linspace(0,1000, interpolated_centerline.shape[0]), interpolated_centerline[:, 1], color='blue', label='Interpolated Centerline')
    plt.title('2D Plot - Differences')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage:
original_centerline = load_centerline(get_base_file_path('saves/centerline.pkl'))
interpolated_centerline = interpolate_and_smooth(original_centerline)
visualize_differences(original_centerline, interpolated_centerline, get_base_file_path('images/track_edited.jpg'))
