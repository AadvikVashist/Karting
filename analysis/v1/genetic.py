import numpy as np
from cost_function import memoized_optimized_cost_function, add_line
import cv2
import time
import pickle as pkl
import os
import utils
from path_manipulation import create_random_path, smooth_centerline
from path_manipulation import mutate
def crossover(parent1, parent2):
    # Perform crossover between two parents to produce offspring
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def compute_sigma(mutation_rate, temperature, sigma_min=0, sigma_max=1):
    # Assuming scalar is a function of temperature
    sigma_rate = 0.1
    if np.random.rand() < mutation_rate*temperature:
        sigma = sigma_max
    else:
        sigma = sigma_min
    return sigma


def evaluate_fitness(paths, kart, track_graph):
    fitness = []
    for path in paths:
        cost, _ = memoized_optimized_cost_function(kart, path, track_graph)
        fitness.append(1 / cost)  # Inverse of cost as fitness
    return np.array(fitness)

def select_parents(paths, fitness):
    # Select parents for mating using roulette wheel selection
    probabilities = fitness / np.sum(fitness)
    indices = np.random.choice(np.arange(len(paths)), size=2, p=probabilities)
    return paths[indices[0]], paths[indices[1]]


def genetic_algorithm(kart, centerline, track_line, track_width, track_graph, track_img, smoothen, initial_temp=1, min_temp=0.05, num_generations=100, population_size=100, mutation_rate=0.1, show=False):
    # Initialize parameters
    # Assuming each point is represented as (x, y) and track_graph has nodes that represent valid positions
    population = [smooth_centerline(create_random_path(track_line, track_width, centerline=centerline, track_graph = track_graph, weight=initial_temp), sigma=5) for _ in range(population_size)]
    temperature = initial_temp  # Initial temperature
    cooling_rate = (min_temp/temperature)**(1/num_generations)  # Cooling rate

    # Evaluate initial fitness
    fitness = evaluate_fitness(population, kart, track_graph)
    best_index = np.argmin(fitness)
    best_path = population[best_index]
    best_cost = 1 / fitness[best_index]
    print(f"Initial Best Cost = {best_cost:,.2f}")

    # Main loop
    x = 0
    while temperature > min_temp:  # Stopping criterion (you can adjust this)
        new_population = []
        x+=1
        start = time.time()
        if population_size < 25:
            gaussian_weights = [mutation_rate] * len(population)
        else:
            lin = np.linspace(-1, 0, len(population))
            gaussian_weights = np.exp(-0.5 * (lin ** 2) / (0.2 ** 2))*mutation_rate   # Adjust the 0.25 std dev as needed
        # plt.plot(x, gaussian_weights)
        # plt.show()
        for parent_index, _ in enumerate(population):
            # Randomly select two parents
            parent1, parent2 = select_parents(population, fitness)

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutate children
            mutation_magnitude = temperature*(mutation_rate**(1/10))*5

            child1 = mutate(centerline, child1, gaussian_weights[parent_index], mutation_magnitude, track_width, track_graph, smoothen=smoothen)
            child2 = mutate(centerline, child2, gaussian_weights[parent_index], mutation_magnitude, track_width, track_graph, smoothen=smoothen)
            # Evaluate fitness of children
            child1_fitness = 1 / memoized_optimized_cost_function(kart, child1, track_graph)[0] #fast
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
        fitness = evaluate_fitness(population, kart, track_graph)
        best_index = np.argmin(fitness)
        best_path = population[best_index]
        best_cost = 1 / fitness[best_index]
        print(f"Gen {x} Best Cost =  {'{:,.2f}'.format(best_cost)} at Temperature = {np.around(temperature,3)} in {time.time() - start} seconds")

        # Draw best path on track image
        # best_path_show = np.append(best_path, [best_path[0]], axis=0)
        best_path_show = best_path
        track_with_path = add_line(track_img.copy(), best_path_show)
        cv2.line(track_with_path, (int(best_path_show[0][0]), int(best_path_show[0][1])), (int(best_path_show[-1][0]), int(best_path_show[-1][1])), (0, 0, 255), 3)
        cv2.putText(track_with_path, f"Cost: {best_cost:,.2f} at Temperature = {np.around(temperature,3)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 50, 0), 2)
        if show:
            cv2.imshow("Best Path", track_with_path)
            cv2.waitKey(1)

        # save this image
        cv2.imwrite(utils.get_base_file_path(os.path.join("images/gens",str(x) + ".png" )), track_with_path)
        pkl.dump(best_path, open(utils.get_base_file_path("saves/temp.pkl"), 'wb'))
        cv2.destroyAllWindows()
    print(f"Final Best Cost = {best_cost:,.2f} at Temperature = {np.around(temperature,3)}")
    return best_path

