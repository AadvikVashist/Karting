import numpy as np
import cv2
import pickle as pkl
from cost_function import optimized_cost_function, add_line
import os
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d
from collections import OrderedDict
import utils



# Define a cache for the optimized_cost_function results

cost_function_cache = OrderedDict()
CACHE_SIZE_LIMIT = 1000

def memoized_optimized_cost_function(kart, path, track_graph):
    # Use tuple of parameters as key to check the cache
    cache_key = (tuple(map(tuple, path)), tuple(track_graph))
    
    if cache_key in cost_function_cache:
        return cost_function_cache.pop(cache_key)
    
    # Compute the result if not in cache
    cost = optimized_cost_function(kart, path, track_graph)
    cost_function_cache[cache_key] = cost
    
    # Move the recently used item to the end of the cache
    cost_function_cache.move_to_end(cache_key)

    # Clear cache entries if the size exceeds the limit
    if len(cost_function_cache) > CACHE_SIZE_LIMIT:
        cost_function_cache.popitem(last=False)
    
    return cost


def resample_centerline(centerline, track_width, num_points):
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

def smooth_centerline(centerline, sigma=2):
    x = [point[0] for point in centerline]
    y = [point[1] for point in centerline]
    x = gaussian_filter(x, sigma)
    y = gaussian_filter(y, sigma)
    return np.column_stack((x, y))



def update_point(point, point_b4, point_aft, weight=0.3):
    # Update a point based on the points before and after it
    direction = point_aft - point_b4
    direction = [direction[1], -direction[0]]
    # check ortho using dot product
    x = np.dot(direction, point_aft - point_b4)
    if abs(x) > 1e-6:
        raise ValueError("The points are not orthogonal")
    direction = direction / np.linalg.norm(direction)
    #ensure that it is clockwise relative to the centerline
    if np.dot(direction, point - point_b4) < 0:
        direction = -direction
    
    new_point = point + direction * weight
    # plt.plot(new_point[0], new_point[1], 'ro', label='Random Path')
    # plt.plot(point[0], point[1], 'bo', label='Centerline')
    # plt.plot(point_b4[0], point_b4[1], 'go', label='previous Path')
    # plt.plot(point_aft[0], point_aft[1], 'o', label='after Path')

    # plt.plot([point[0], new_point[0]], [point[1], new_point[1]])
    # plt.plot([point_b4[0], point_aft[0]], [point_b4[1], point_aft[1]])
    # plt.legend()
    # plt.show()
    return new_point

def create_random_path(track_line, track_width, weight = 0.3):
    # Create a random path within the track boundaries
    random_paths = []
    #add 0 index to end of centerline array
    for i in range(0, len(track_line)):
        if i >= CONNECTION_BUFFER and i < len(track_line) - CONNECTION_BUFFER:

            random_offset = np.random.uniform(-track_width[i] * weight, track_width[i] * weight)
            # get the point before, and after the current point in the centerline to get the direction
            # get the vector between the two points
            random_path_point = update_point(track_line[i], track_line[i - 1], track_line[i + 1], random_offset)
            random_paths.append(random_path_point)
            # plt.plot(random_path_point[0], random_path_point[1], 'ro', label='Random Path')
            # plt.plot(centerline[i][0], centerline[i][1], 'bo', label='Centerline')
            # plt.plot(centerline[i - 1][0], centerline[i - 1][1], 'go', label='previous Path')
            # plt.plot(centerline[i + 1][0], centerline[i + 1][1], 'o', label='after Path')
            # plt.legend()
            # plt.show()
        else:
            random_paths.append(track_line[i])    
    # cv2.imshow("Random Path", add_line(track_img.copy(), random_paths))
    # cv2.waitKey(1)
    return np.array(random_paths)



def crossover(parent1, parent2):
    # Perform crossover between two parents to produce offspring
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def compute_sigma(mutation_rate, temperature, sigma_min=0, sigma_max=1):
    # Assuming scalar is a function of temperature

    # Compute the probability factor based on mutation rate and scalar
    # weight_mutation = mutation_rate ** (1.0/3.0)
    # weight_temperature = temperature

    # # Adjust the sigma value based on mutation rate, temperature, and a random factor
    # sigma = sigma_min + ((sigma_max - sigma_min) * np.random.rand())**(1/(weight_mutation * weight_temperature))
    if np.random.rand() < SIGMA_RANDOM:
        sigma = sigma_max
    else:
        sigma = sigma_min
    
    # sigma = sigma_min + (sigma_max - sigma_min) * (np.random.rand())**10
    # # sigma = sigma_min + (sigma_max - sigma_min) * (np.random.rand())**(1/2)
    # if sigma > sigma_max:
    #     sigma = sigma_max
    return sigma

def mutate_old(centerline, path, mutation_rate, track_width, temperature):
    # Apply mutation to a path
    scalar = temperature ** 2
    sigma = np.round(compute_sigma(mutation_rate, temperature, sigma_max=SIGMA_MAX))
    i = 0
    while i < len(path):
        if i <= CONNECTION_BUFFER or i >= len(path) - CONNECTION_BUFFER:
            i+=1
            continue
        if np.random.rand() < mutation_rate:
            random_ret = np.random.uniform(-1*scalar/5, scalar/5)
            quant = int(np.round(np.random.uniform(low = MUTATION_SHIFT_MIN, high = MUTATION_SIZE_MAX)))
            if i + quant + 1 > len(path):
                quant = len(path) - i-2
            shift = [update_point(path[iter_var], centerline[iter_var-1], path[iter_var+1], random_ret*track_width[iter_var]) for iter_var in range(i, i+quant)]
            path[i:i+quant] = shift
            i+=quant
        i+=1
    path = smooth_centerline(path, sigma)
    return path


def mutate(centerline, path, mutation_rate, track_width, temperature):
    # Apply mutation to a path
    scalar = temperature ** 2
    sigma = np.round(compute_sigma(mutation_rate, temperature, sigma_max=SIGMA_MAX))
    i = 0
    path = np.array(path) 
    path_appended = np.vstack((path[-CONNECTION_BUFFER::], path, path[0:CONNECTION_BUFFER])) #append the first x points to the end of the array
    #do the same for the track_width and centerline
    centerline_appended = np.vstack((centerline[-CONNECTION_BUFFER::], centerline, centerline[0:CONNECTION_BUFFER]))
    width_appended = np.hstack((track_width[-CONNECTION_BUFFER::], track_width, track_width[0:CONNECTION_BUFFER]))
    track_width = np.array(track_width)
    while i < len(path_appended):
        if i < 1 or i > len(path_appended) - 2:
            i+=1
            continue
        elif np.random.rand() < mutation_rate:
            random_ret = np.random.uniform(-1*scalar/5, scalar/5)
            quant = int(np.round(np.random.uniform(low = MUTATION_SHIFT_MIN, high = MUTATION_SIZE_MAX)))
            
            if i + quant >= len(path_appended):
                quant = len(path_appended) - i-1
            try:
                shift = [update_point(path_appended[iter_var], centerline_appended[iter_var-1], path_appended[iter_var+1], random_ret*width_appended[iter_var]) for iter_var in range(i, i+quant)]
            except IndexError as e:
                print(i, i+quant, len(path_appended))
            path_appended[i:i+quant] = shift
            i+=quant
        i+=1
    path_appended = smooth_centerline(path_appended, sigma)
    #take the average between the appended 5 points and the original points
    connect_ends = np.mean([path_appended[0:CONNECTION_BUFFER*2], path_appended[len(path)::]],axis=0)
    ret_path = np.vstack((connect_ends[-CONNECTION_BUFFER:], path_appended[2*CONNECTION_BUFFER:-CONNECTION_BUFFER*2], connect_ends[0:CONNECTION_BUFFER] ))


    return ret_path

def evaluate_fitness(paths, kart, track_graph, track_img):
    fitness = []
    for path in paths:
        cost = memoized_optimized_cost_function(kart, path, track_graph)
        fitness.append(1 / cost)  # Inverse of cost as fitness
    return np.array(fitness)

def select_parents(paths, fitness):
    # Select parents for mating using roulette wheel selection
    probabilities = fitness / np.sum(fitness)
    indices = np.random.choice(np.arange(len(paths)), size=2, p=probabilities)
    return paths[indices[0]], paths[indices[1]]


def genetic_algorithm(kart, centerline, track_line, track_width, track_graph, track_img):
    # Initialize parameters

    population = [smooth_centerline(create_random_path(track_line, track_width, weight=INITIAL_TEMPERATURE)) for _ in range(POPULATION_SIZE)]
    temperature = INITIAL_TEMPERATURE  # Initial temperature
    
    cooling_rate = (MIN_TEMP/1.01)**(1/NUM_GENERATIONS)

    # Evaluate initial fitness
    fitness = evaluate_fitness(population, kart, track_graph, track_img)
    best_index = np.argmin(fitness)
    best_path = population[best_index]
    best_cost = 1 / fitness[best_index]
    print(f"Initial Best Cost = {best_cost:,.2f}")

    # Main loop
    x = 0
    while temperature > MIN_TEMP:  # Stopping criterion (you can adjust this)
        new_population = []
        x+=1
        start = time.time()
        for parent_index, _ in enumerate(population):
            # Randomly select two parents
            parent1, parent2 = select_parents(population, fitness)

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutate children
            child1 = mutate(centerline, child1, MUTATION_RATE, track_width, temperature)
            child2 = mutate(centerline, child2, MUTATION_RATE, track_width, temperature)
            # Evaluate fitness of children
            child1_fitness = 1 / memoized_optimized_cost_function(kart, child1, track_graph) #fast
            # child2_fitness = 1 / memoized_optimized_cost_function(kart, child2, track_graph) #fast

            # Select one child based on fitness and temperature
            if np.random.rand() < np.exp((fitness[parent_index] - child1_fitness) / temperature):
                new_population.append(child1)
            else:
                new_population.append(child2)

        # Update population and temperature
        population = new_population
        temperature *= cooling_rate

        # Find and display the best path
        fitness = evaluate_fitness(population, kart, track_graph, track_img)
        best_index = np.argmin(fitness)
        best_path = population[best_index]
        best_cost = 1 / fitness[best_index]
        print(f"Gen {x} Best Cost =  {'{:,.2f}'.format(best_cost)} at Temperature = {temperature} in {time.time() - start} seconds")

        # Draw best path on track image
        # best_path_show = np.append(best_path, [best_path[0]], axis=0)
        best_path_show = best_path
        track_with_path = add_line(track_img.copy(), best_path_show)
        cv2.line(track_with_path, (int(best_path_show[0][0]), int(best_path_show[0][1])), (int(best_path_show[-1][0]), int(best_path_show[-1][1])), (0, 0, 255), 3)
        cv2.putText(track_with_path, f"Cost: {best_cost:,.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 50, 0), 2)
        if SHOW:
            cv2.imshow("Best Path", track_with_path)
            cv2.waitKey(1)

        # save this image
        cv2.imwrite(utils.get_base_file_path(os.path.join("images/gens",str(x) + ".png" )), track_with_path)
        pkl.dump(best_path, open(utils.get_base_file_path("saves/temp.pkl"), 'wb'))
        cv2.destroyAllWindows()
    print(f"Final Best Cost = {best_cost:,.2f}")
    return best_path

# Example usage

# main.py
from kart import Kart
import numpy as np
import pickle as pkl
import utils as utils

def train(track_line):
    if os.path.exists(utils.get_base_file_path('images/gens')):
        #move the files to a new folder called images/old_gens/{time}
        new = utils.get_base_file_path('images/old_gens/' + str(time.time()))
        os.makedirs(new)
        os.rename(utils.get_base_file_path('images/gens'), new)
        os.makedirs(utils.get_base_file_path('images/gens'))
    else:
        os.makedirs(utils.get_base_file_path('images/gens'))
    centerline = np.array(pkl.load(open(utils.get_base_file_path('saves/rearranged_centerline.pkl'), 'rb')))
    track_width = np.array(pkl.load(open(utils.get_base_file_path('saves/track_width.pkl'), 'rb')))

    contours = pkl.load(open(utils.get_base_file_path('saves/track_contours.pkl'), 'rb'))
    track_graph = pkl.load(open(utils.get_base_file_path('saves/graph.pkl'), 'rb'))
    track_img = cv2.imread(utils.get_base_file_path('images/track_edited.jpg'))
    # Create Kart instance

    kart = Kart(
        max_speed=130,           # 13 m/s
        acceleration=10,         # 1 m/s²
        braking=40 ,              # 4 m/s²
        turning_radius=60,       # 4 m
        steerability=1.3,        # Adjust this based on track testing
        width=14,                # 1.4 m
        length=23                # 2.3 m
    )

    new_centerline, new_track_width = resample_centerline(centerline, track_width, POINTS)
    new_trackline, _ = resample_centerline(track_line, range(track_line.shape[0]), POINTS)
    best_path = genetic_algorithm(kart, new_centerline, new_trackline, new_track_width, track_graph, track_img)
    # Save best path
    pkl.dump(best_path, open(utils.get_base_file_path('saves/best_path.pkl'), 'wb'))

def fresh_run():
    track_line = np.array(pkl.load(open(utils.get_base_file_path('saves/rearranged_centerline.pkl'), 'rb')))
    train(track_line)
def run_prev():
    track_line = np.array(pkl.load(open(utils.get_base_file_path('saves/temp.pkl'), 'rb')))
    train(track_line)

# Define genetic algorithm parameters
POPULATION_SIZE = 50
MUTATION_RATE = 0.005

CONNECTION_BUFFER = 5

INITIAL_TEMPERATURE = 1
MIN_TEMP = 0.1

NUM_GENERATIONS = 1000
POINTS = 1000

MUTATION_SHIFT_MIN = 10
MUTATION_SIZE_MAX = 50

SIGMA_MAX = 1
SIGMA_RANDOM = 0.01

SHOW = True
# Load centerline and other track data

improved_line = np.array(pkl.load(open(utils.get_base_file_path('saves/best_path_2.pkl'), 'rb')))

fresh_run()