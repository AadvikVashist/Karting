import numpy as np
import cv2
import pickle as pkl
import os
import time
import utils
from kart import Kart
from path_manipulation import resample_centerline
from genetic import genetic_algorithm

def save_best_path(dir,file_name, best_path):
    # get the number of files in the dir that contain that name
    #get raw file name without extension
    file = file_name.split('.')[0]
    num_files = len([f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and file in f])
    file_name = file + str(num_files) + "." + file_name.split('.')[1]
    pkl.dump(best_path, open(os.path.join(dir, file_name), 'wb'))

def get_track_line(contours, track_img= None):
    contours  = [contours[1].squeeze(), contours[0].squeeze()]   
    track_center = np.mean(np.vstack(contours), axis=0)
    if track_img is not None:
        cv2.polylines(track_img, [contours[0].astype(int)], isClosed=True, color=(0, 0, 255), thickness=3)
        cv2.polylines(track_img, [contours[1].astype(int)], isClosed=True, color=(0, 255, 0), thickness=3)
        cv2.circle(track_img, (int(track_center[0]), int(track_center[1])), 5, (0, 0, 255), -1)
        cv2.imshow("Track", track_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return track_center, contours
    
def train(kart, track_line):
    if os.path.exists(utils.get_base_file_path('images/gens')):
        #move the files to a new folder called images/old_gens/{time}
        new = utils.get_base_file_path('images/old_gens/' + str(time.time()))
        os.makedirs(new)
        os.rename(utils.get_base_file_path('images/gens'), new)
        os.makedirs(utils.get_base_file_path('images/gens'))
    else:
        os.makedirs(utils.get_base_file_path('images/gens'))
        
    if not os.path.exists(utils.get_base_file_path('saves/gens')):
        os.makedirs(utils.get_base_file_path('saves/gens'))
    centerline = np.array(pkl.load(open(utils.get_base_file_path('saves/rearranged_centerline.pkl'), 'rb')))
    track_width = np.array(pkl.load(open(utils.get_base_file_path('saves/track_width.pkl'), 'rb')))

    track_graph = pkl.load(open(utils.get_base_file_path('saves/graph.pkl'), 'rb'))
    track_img = cv2.imread(utils.get_base_file_path('images/track_edited.jpg'))
    contours = pkl.load(open(utils.get_base_file_path('saves/track_contours.pkl'), 'rb'))
    track_center, contours = get_track_line(contours)
    inner_contour, outer_contour = contours
    new_centerline, new_track_width = resample_centerline(centerline, track_width, POINTS)
    new_trackline, _ = resample_centerline(track_line, range(track_line.shape[0]), POINTS)

    best_path = genetic_algorithm(kart, new_centerline, new_trackline, new_track_width, track_graph, track_img, SMOOTHEN, initial_temp=INITIAL_TEMPERATURE, min_temp=MIN_TEMP, num_generations=NUM_GENERATIONS, population_size=POPULATION_SIZE, mutation_rate=MUTATION_RATE, show=SHOW)


    # Save best path
    save_best_path(utils.get_base_file_path('saves/gens'), 'best_path.pkl', best_path)
    pkl.dump(best_path, open(utils.get_base_file_path('saves/best_path.pkl'), 'wb'))

def fresh_run(kart):
    track_line = np.array(pkl.load(open(utils.get_base_file_path('saves/rearranged_centerline.pkl'), 'rb')))
    train(kart,track_line)
def run_prev(kart):
    track_line = np.array(pkl.load(open(utils.get_base_file_path('saves/temp.pkl'), 'rb')))
    train(kart,track_line)


# Define genetic algorithm parameters
POPULATION_SIZE = 15
NUM_GENERATIONS = 400

INITIAL_TEMPERATURE = 0.9
MIN_TEMP = 0.2

MUTATION_RATE = 0.01

POINTS = 1000
MUTATION_MAG = 1/10

SHOW = True
# USE_WIDTH = True
SMOOTHEN = True
kart = Kart(
    max_speed=130,           # 13 m/s
    acceleration=10,         # 1 m/s²
    braking=40 ,              # 4 m/s²
    turning_radius=60,       # 4 m
    steerability=1.3,        # Adjust this based on track testing
    width=14,                # 1.4 m
    length=23                # 2.3 m
)

fresh_run(kart)
# run_prev(kart)