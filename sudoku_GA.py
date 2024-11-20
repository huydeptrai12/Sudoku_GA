import numpy as np
import random
import time

# Parameters for the genetic algorithm
POPULATION_SIZE = 500
MUTATION_RATE = 0.1
GENERATIONS = 1000
ELITISM_COUNT = 5

# Helper functions
def create_individual(puzzle):
    """
        Initialize the board 
        FOR EACH ROW OF THE BOARD, IT WILL BE FILLED WITH RANDOM VALUE SO THAT EACH ROW DOESN'T HAVE ANY DUPLICATE VALUES
    """
    individual = []
    for row in puzzle:
        new_row = np.array(row)
        # FIND INDEX THAT CONTAINS EMPTY VALUE
        empty_indices = [i for i in range(9) if new_row[i] == 0]
        # FIND ALL THE POSSIBLE VALUES EXCEPT THOSE FIXED VALUES
        possible_values = list(set(range(1, 10)) - set(new_row))
        # SHUFFLE TO POSSIBLE VALUES
        random.shuffle(possible_values)
        for i, index in enumerate(empty_indices):
            new_row[index] = possible_values[i]
        individual.append(new_row)
    return np.array(individual)

def initialize_population(puzzle):
    """Initialize population with random individuals that fit the puzzle constraints."""
    return [create_individual(puzzle) for _ in range(POPULATION_SIZE)]

def fitness(individual):
    """Calculate fitness based on unique values in columns and subgrids."""
    col_score = sum(len(np.unique(individual[:, col])) for col in range(9))
    subgrid_score = sum(len(np.unique(individual[row:row+3, col:col+3])) 
                        for row in range(0, 9, 3) for col in range(0, 9, 3))
    # All the row in the board doesn't have any duplicated values so the row_score will always be 81
    row_score = 81
    return col_score + subgrid_score + row_score

def selection(population, fitness_scores):
    """Select individuals based on fitness scores."""
    return random.choices(population, weights=fitness_scores, k=2)

# Crossover methods that preserve row uniqueness
def crossover_row_based(parent1, parent2):
    """Row-based crossover: randomly select rows from each parent."""
    child = np.zeros_like(parent1)
    for row in range(9):
        child[row, :] = parent1[row, :] if random.random() < 0.5 else parent2[row, :]
    return child

def crossover_one_point(parent1, parent2):
    """One-point crossover: split parents at a random row."""
    point = random.randint(1, 8)
    child = np.vstack((parent1[:point], parent2[point:]))
    return child

def crossover_two_point(parent1, parent2):
    """Two-point crossover: split parents at two random rows."""
    point1, point2 = sorted(random.sample(range(1, 9), 2))
    child = np.vstack((parent1[:point1], parent2[point1:point2], parent1[point2:]))
    return child

# Mutation methods that preserve row uniqueness
def mutation_swap(individual, puzzle):
    """Swap two random values in the same row, preserving row uniqueness."""
    for row in range(9):
        empty_indices = [col for col in range(9) if puzzle[row][col] == 0]
        if len(empty_indices) > 1 and random.random() < MUTATION_RATE:
            i, j = random.sample(empty_indices, 2)
            individual[row, i], individual[row, j] = individual[row, j], individual[row, i]
    return individual

def genetic_algorithm(puzzle, crossover_method, mutation_method):
    """Main function to run the genetic algorithm for solving Sudoku."""
    population = initialize_population(puzzle)
    start_time = time.time()
    
    for generation in range(GENERATIONS):
        
        # Calculate fitness scores
        fitness_scores = [fitness(ind) for ind in population]
        max_fitness = max(fitness_scores)
        best_individual = population[fitness_scores.index(max_fitness)]
        
        # Check if solution is found
        if max_fitness == 243:  # 9 columns + 9 subgrids all max out at 9 unique values each
            print(f"Solved in generation {generation}", flush = True)
            print(f"Time taken: {time.time() - start_time:.2f} seconds")
            return best_individual, max_fitness
        else:
            print("THE ALGORITHM IS RUNNING", end = '\r')
        # Sort population by fitness for elitism
        sorted_population = sorted(population, key=fitness, reverse=True)
        
        # Create new population with elitism
        new_population = sorted_population[:ELITISM_COUNT]
        
        # Generate rest of the population using crossover and mutation
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = selection(population, fitness_scores)
            child = crossover_method(parent1, parent2)
            child = mutation_method(child, puzzle)
            new_population.append(child)
        
        population = new_population

    print("No solution found within the generation limit.")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    return best_individual, max_fitness

def calculate_num_of_blanks(board):
    """CALCULATE THE NUMBER OF BLANKS IN THE BOARD"""
    unique, counts = np.unique(board, return_counts=True)
    return counts[0]


