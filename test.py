from sudoku_GA import *
from sudoku import Sudoku
import sys

if __name__ == "__main__":
    try:
        difficulty = float(sys.argv[1])
    except:
        difficulty = 0.3
        
    print(difficulty)
    sudoku = Sudoku(3).difficulty(difficulty)  # Standard Sudoku with 3x3 subgrids
    puzzle_board = sudoku.board  # Get the generated puzzle as a list of lists
    puzzle_array = np.array(puzzle_board)  # Convert to NumPy array for our algorithm
    puzzle_array = np.where(puzzle_array == None, 0, puzzle_array)
    # Helper function to display Sudoku
    def print_sudoku(title, board):
        """Display the Sudoku grid with a title using the Sudoku module."""
        print(f"\n{title}")
        sudoku = Sudoku(3)
        sudoku.board = board.tolist() if isinstance(board, np.ndarray) else board
        sudoku.show()

    # Fitness function and other genetic algorithm components remain unchanged
    print(calculate_num_of_blanks(puzzle_array))
    # Testing the algorithm
    print_sudoku("Initial Sudoku Puzzle", puzzle_board)

    # Solve the Sudoku using our genetic algorithm
    solution, score = genetic_algorithm(puzzle_array, crossover_row_based, mutation_swap)

    # Display the solutions
    print_sudoku("Our Genetic Algorithm Solution", solution)

    # Solve the Sudoku using the Sudoku module's built-in solver
    solved_board = sudoku.solve().board
    print_sudoku("Sudoku Module Solution", solved_board)
    print(f"SCORE {score}")
    # Compare solutions
    if np.array_equal(solution, np.array(solved_board)):
        print("\nCORRECT")
    else:
        print("\nFALSE")