import utils
import cv2
import numpy as np
import pickle as pkl
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        params['click_point'] = (x, y)

def display_image_and_capture_click(img):
    cv2.namedWindow("Image")
    click_params = {'click_point': None}
    cv2.setMouseCallback("Image", click_event, click_params)
    cv2.imshow("Image", img)
    cv2.waitKey(0)  # Wait until a click or key press
    cv2.destroyAllWindows()
    return click_params['click_point']

def find_closest_contour_point(contour, point):
    distances = np.sqrt((contour[:, 0] - point[0])**2 + (contour[:, 1] - point[1])**2)
    min_index = np.argmin(distances)
    return min_index

def rearrange_contour(contour, start_index):
    return np.roll(contour, -start_index, axis=0)

def process_image_contour(img, contour):
    # Display the image with contour
    img_with_contour = img.copy()
    for idx, point in enumerate(contour):
        if idx == len(contour) - 1:
            continue
            # idx = -1
        cv2.line(img_with_contour, contour[idx].astype(int), contour[(idx + 1)].astype(int), (0, 255, 0), 2)
    
    # Get user click
    clicked_point = display_image_and_capture_click(img_with_contour)
    if clicked_point is None:
        print("No point was clicked.")
        return
    
    # Find closest point on contour to the clicked point
    closest_index = find_closest_contour_point(contour, clicked_point)
    
    # Rearrange contour to start from the closest point and go CCW
    rearranged_contour = rearrange_contour(contour, closest_index)
    
    # Show the rearranged contour
    img_with_rearranged_contour = img.copy()
    for idx, point in enumerate(rearranged_contour):
        if idx == len(contour) - 1:
            continue
            # idx = -1
        cv2.line(img_with_rearranged_contour, rearranged_contour[idx].astype(int), rearranged_contour[(idx + 1)].astype(int), (0, 255, 0), 2)
    cv2.imshow("Rearranged Contour", img_with_rearranged_contour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return rearranged_contour

# Example usage
# Load an image using cv2
# img = cv2.imread('path_to_image.jpg')
# contour = np.array([[[10, 10]], [[100, 100]], [[200, 50]], [[50, 200]]])
# process_image_contour(img, contour)



image = cv2.imread(utils.get_base_file_path('images/track_edited.jpg'), cv2.IMREAD_GRAYSCALE)
contour = np.array(pkl.load(open(utils.get_base_file_path('saves/centerline.pkl'), 'rb')))
new_contour = process_image_contour(image, contour)
# Save the new contour
pkl.dump(new_contour, open(utils.get_base_file_path('saves/rearranged_centerline.pkl'), 'wb'))
