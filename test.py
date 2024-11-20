from sudoku_GA import *
# This module is used for generating sudoku tables based on the specified difficulty 
# and quickly find the true solution used for comparing with our solution
from sudoku import Sudoku
import sys

# Helper function to display Sudoku
def print_sudoku(title, board):
    # DISPLAY THE SUDOKU BOARD
    print(f"\n{title}")
    sudoku = Sudoku(3)
    sudoku.board = board.tolist() if isinstance(board, np.ndarray) else board
    sudoku.show()

if __name__ == "__main__":
    # SET THE DIFFICULTY 
    try:
        # GET THE DIFFICULTY FROM THE TERMINAL
        difficulty = float(sys.argv[1])
    except:
        # DEFAULT DIFFICULTY
        difficulty = 0.3
        
    if (difficulty <= 0 or difficulty > 1):
        # CHECK IF DIFFICULTY IS VALID IN RANGE (0, 1]
        print("DIFFICULTY INVALID, THE DEFAULT VALUE WILL BE USED")
        difficulty = 0.3

    print(f"DIFFICULTY {difficulty}")
    sudoku = Sudoku(3).difficulty(difficulty)  # GET A SUDOKU TABLE WITH RESPECT TO THE DIFFICUTY
    puzzle_board = sudoku.board  # GET THE BOARD 
    puzzle_array = np.array(puzzle_board)  # CONVERT TO NUMPY ARRAY FOR OUT ALGORITHM
    puzzle_array = np.where(puzzle_array == None, 0, puzzle_array) # REPLACE ANY NONE VALUE WITH 0

    # CALCULATE THE NUMBER OF BLANKS IN THE BOARD
    print(f"NUMBER OF BLANKS {calculate_num_of_blanks(puzzle_array)}")
    # SHOW THE INITIAL BOARD
    print_sudoku("Initial Sudoku Puzzle", puzzle_board)

    # Solve the Sudoku using the Sudoku module's built-in solver
    solved_board = sudoku.solve().board
    print_sudoku("Sudoku Module Solution", solved_board)

    # SOLVE THE SUDOKU USING OUR ALGORITHM
    solution, score = genetic_algorithm(puzzle_array, crossover_row_based, mutation_swap)

    # PRINT THE SOLUTION
    print_sudoku("Our Genetic Algorithm Solution", solution)
    print(f"GA ALGORITHM SCORE {score}")

    # Compare solutions
    if np.array_equal(solution, np.array(solved_board)):
        print("OUR SOLUTION IS CORRECT")
    else:
        print("OUR SOLUTION IS FALSE")