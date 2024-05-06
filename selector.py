import cv2
def select_points(event, x, y, flags, param):
    global start_point, end_point
    if event == cv2.EVENT_LBUTTONDOWN:
        if start_point is None:
            start_point = (y, x)
            print(f"Start point selected: {start_point}")
        elif end_point is None:
            end_point = (y, x)
            print(f"End point selected: {end_point}")
        else:
            print("Both points are already selected.")


    