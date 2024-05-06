import cv2
import numpy as np

# Load the image in grayscale
img = cv2.imread('./track_edited.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to smooth the image
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Edge detection using Canny
edges = cv2.Canny(blurred, 50, 150)

# Apply dilation followed by erosion to close gaps in edges
kernel = np.ones((7, 7), np.uint8)
closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
for i in range(5):
    closed_edges = cv2.dilate(closed_edges, kernel, iterations=2)
    closed_edges = cv2.erode(closed_edges, kernel, iterations=2)

closed_edges = cv2.erode(closed_edges, kernel, iterations=1)

# Find contours in the closed edges
contours, _ = cv2.findContours(closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank image to draw contours
output_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
# Sort contours by length in descending order
contours_sorted = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)

# Keep only the two longest contours
top_contours = contours_sorted[:2]

# Create a blank image to draw contours
output_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

# Draw the two longest contours
for contour in top_contours:
    cv2.drawContours(output_img, [contour], -1, (255, 255, 255), 2)
# Show the original and processed images
cv2.imshow('Original Image', img)
cv2.waitKey(0)
cv2.imshow('Original Image', edges)
cv2.waitKey(0)
cv2.imshow('Original Image', closed_edges)
cv2.waitKey(0)
cv2.imshow('Detected Contours', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# save the output_img
cv2.imwrite('output_img.jpg', output_img)
import pickle as pkl
pkl.dump(top_contours, open('output.pkl', 'wb'))


