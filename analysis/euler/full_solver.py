# main.py
from kart import Kart
from integrated_approach import integrated_approach
import numpy as np
import pickle as pkl
import utils as utils
# Load centerline and other track data
centerline = pkl.load(open(utils.get_base_file_path('saves/centerline.pkl'), 'rb'))
track_width = pkl.load(open(utils.get_base_file_path('saves/track_width.pkl'), 'rb'))
contours = pkl.load(open(utils.get_base_file_path('saves/track_contours.pkl'), 'rb'))
# Create Kart instance
kart = Kart(
    max_speed=130,           # 13 m/s
    acceleration=20,         # 2 m/s²
    braking=40 ,              # 4 m/s²
    turning_radius=40,       # 4 m
    steerability=1.3,        # Adjust this based on track testing
    width=14,                # 1.4 m
    length=23                # 2.3 m
)

# Optimize racing line
optimized_line = integrated_approach(centerline, track_width, contours, kart)

# Visualize or further process optimized_line as needed
