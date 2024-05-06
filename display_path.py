import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import cv2

improved_path = pkl.load(open('improved_path.pkl', 'rb'))
output_img = cv2.imread('track_edited.jpg')
output_img = np.zeros_like(output_img)
for i in range(len(improved_path) - 1):
    cv2.line(output_img, (int(improved_path[i][0]), int(improved_path[i][1])), (int(improved_path[i + 1][0]), int(improved_path[i + 1][1])), (0, 255, 0), 1)
cv2.imshow('Improved Path', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

