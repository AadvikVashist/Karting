import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import pickle as pkl
from kart import Kart
from  cost_function import add_line, cost_function
import numpy as np
import random
import matplotlib.pyplot as plt
def modify_path(path):
    # Implement logic to modify the path slightly
    # For example, shift each point slightly based on random noise or a deterministic strategy
    new_path = path.copy()
    
    # Example: Move each point slightly randomly
    diff = np.random.randint(-20, 20, size=(len(new_path), 2))
    for i in range(1, len(new_path) - 1):
        new_path[i] = (new_path[i][0] + diff[i][0], new_path[i][1] + diff[i][1])
        # new_path[i] = (new_path[i][0] + np.random.randint(-50, 50), new_path[i][1] + np.random.randint(-50, 50))
    # import matplotlib.pyplot as plt
    # gauss_path = gaussian_filter(new_path, 3)
    new_path = np.array(new_path)
    x = new_path[:, 0]
    y= new_path[:, 1]
    gauss_path = np.column_stack((gaussian_filter(x, 3), gaussian_filter(y, 3)))
    # add the first point to the end
    gauss_path = np.vstack((gauss_path, gauss_path[0]))
    
    return gauss_path



def crossover(parent1, parent2):
    split = random.randint(1, len(parent1) - 2)
    return np.vstack((parent1[:split], parent2[split:]))


def initialize_population(centerline, pop_size=50):
    population = []
    for _ in range(pop_size):
        new_path = modify_path(centerline)
        # Further randomize initial paths
        if random.random() < 0.5:
            new_path = improved_mutate(new_path, 0.5)
        population.append(new_path)
    return population

def improved_mutate(path, mutation_rate=0.2, temperature=1.0):
    new_path = path.copy()
    adj_temp = int(50 * temperature)
    for i in range(1, len(path) - 1):
        if random.random() < mutation_rate:
            new_path[i] = (new_path[i][0] + random.randint(-1 * adj_temp, adj_temp), new_path[i][1] + random.randint(-1 * adj_temp, adj_temp))
    return new_path

def improved_genetic_algorithm_with_visualization(kart, centerline, track_graph, track_img, pop_size=50, generations=200, mutation_rate=0.2):
    population = initialize_population(centerline, pop_size)
    best_paths = []

    for gen in range(generations):
        population = sorted(population, key=lambda path: cost_function(kart, path[0], path[-1], path, track_graph))
        population_cost = [cost_function(kart, path[0], path[-1], path, track_graph) for path in population]
        # Display the best path in the current generation
    
        # output_img = track_img.copy()
        # output_img = add_line(output_img, population[0])
        # cv2.imshow('Best Path ' + str(gen), output_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        new_population = population[:pop_size//4]  # Elitism, select the top 25% paths
        
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(population[:pop_size//2], 2)
            child = crossover(parent1, parent2)
            child = improved_mutate(child, mutation_rate, temperature=1.0 - gen / generations)
            new_population.append(child)
            
        population = new_population
        best_path = population[0]
        best_paths.append(best_path)
        best_cost = cost_function(kart, best_path[0], best_path[-1], best_path, track_graph)
        print(f"Generation {gen + 1}, Best Cost: {best_cost}")

    return best_path

# Example usage
kart = Kart(max_speed=30, acceleration=3, braking=5, turning_radius=1.5, width=1.0, length=1.5, steerability=0.5)
track_graph = pkl.load(open('graph.pkl', 'rb'))
track_img = cv2.imread('track_edited.jpg')
centerline = pkl.load(open('interpolated_centerline.pkl', 'rb'))

optimal_path = improved_genetic_algorithm_with_visualization(kart, centerline, track_graph, track_img, pop_size=50, generations=200, mutation_rate=0.9)

# Display the improved path
output_img = add_line(track_img, optimal_path)
cv2.imshow('Improved Path', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
