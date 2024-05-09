import numpy as np
import cv2
from cost_function import CostFunctionCache, calculate_cost, calculate_vectorized_cost
from path_manipulation import resample_path, create_random_path, smooth_path, mutate_path
from kart import Kart
import pickle as pkl
import utils
cache = CostFunctionCache()

def memoized_optimized_cost_function(kart, path, track_graph):
    return cache.get(kart, path, track_graph)

def evaluate_fitness(paths, kart, track_graph, min_dist, curr_track_img):
    fitness = []
    for path in paths:
        cost, collisions = memoized_optimized_cost_function(kart, path, track_graph)
        # Introduce an exponential penalty for collisions to heavily discourage them
        collision_penalty = 10000 * collisions if collisions > 0 else 0
        path_fitness = 1 / (cost + collision_penalty)  # Inverse of the total cost
        fitness.append(path_fitness)
    return np.array(fitness)


def crossover(parent1, parent2):
    # Ensure the parents are numpy arrays for easier manipulation
    parent1, parent2 = np.array(parent1), np.array(parent2)
    # Randomly select a crossover point, avoiding the very start and end of the path
    crossover_point = np.random.randint(1, len(parent1) - 1)

    # Create two new children by combining segments from each parent
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]), axis=0)
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]), axis=0)

    return child1, child2


# Tournament selection process
def tournament_selection(paths, fitness, tournament_size=3):
    selected_indices = []
    for _ in range(2):
        participants = np.random.choice(len(paths), tournament_size, replace=False)
        best_participant = participants[np.argmax(fitness[participants])]
        selected_indices.append(best_participant)
    return paths[selected_indices[0]], paths[selected_indices[1]]

# Convergence criteria
def check_convergence(fitness_history, tolerance=0.01, patience=10):
    if len(fitness_history) > patience:
        recent_improvements = np.diff(fitness_history[-patience:])
        if np.all(np.abs(recent_improvements) < tolerance):
            return True
    return False

# Fitness evaluation with heavy penalties for collisions
def evaluate_fitness(paths, kart, track_graph, min_dist, curr_track_img):
    fitness = []
    for path in paths:
        cost, collisions = memoized_optimized_cost_function(path,kart, track_graph)
        collision_penalty = 10000 * collisions if collisions > 0 else 0
        path_fitness = 1 / (cost + collision_penalty)
        fitness.append(path_fitness)
    return np.array(fitness)

# Genetic algorithm core
def genetic_algorithm(kart, centerline, track_line, track_width, track_graph, track_img, contour, min_dist=0):
    population = [smooth_path(create_random_path(track_line, track_width, centerline=centerline, track_graph=track_graph), sigma=5) for _ in range(POPULATION_SIZE)]
    temperature = INITIAL_TEMPERATURE
    cooling_rate = (MIN_TEMP / temperature) ** (1 / NUM_GENERATIONS)
    fitness_history = []

    for gen in range(NUM_GENERATIONS):
        fitness = evaluate_fitness(population, kart, track_graph, min_dist, track_img)
        best_index = np.argmax(fitness)
        best_path = population[best_index]
        best_cost = 1 / fitness[best_index]
        fitness_history.append(best_cost)

        if check_convergence(fitness_history):
            print(f"Convergence reached after {gen+1} generations.")
            break

        new_population = []
        for _ in range(len(population) // 2):
            parent1, parent2 = tournament_selection(population, fitness)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate_path(child1)
            child2 = mutate_path(child2)
            new_population.extend([child1, child2])

        population = new_population
        temperature *= cooling_rate
        print(f"Generation {gen+1}: Best Cost = {best_cost:.4f}")

    return best_path

# Parameters for the genetic algorithm
POPULATION_SIZE = 50
INITIAL_TEMPERATURE = 1.0
MIN_TEMP = 0.05
NUM_GENERATIONS = 100

# Load your data and run the genetic algorithm

centerline = np.array(pkl.load(open(utils.get_base_file_path('saves/rearranged_centerline.pkl'), 'rb')))
track_width = np.array(pkl.load(open(utils.get_base_file_path('saves/track_width.pkl'), 'rb')))
track_graph = pkl.load(open(utils.get_base_file_path('saves/graph.pkl'), 'rb'))
track_img = cv2.imread(utils.get_base_file_path('images/track_edited.jpg'))

kart = Kart(max_speed=130, acceleration=10, braking=40, turning_radius=60, steerability=1.3)
best_path = genetic_algorithm(kart, centerline, centerline, track_width, track_graph, track_img, None, 0)
