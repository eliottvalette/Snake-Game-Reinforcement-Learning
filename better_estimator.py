import numpy as np
from itertools import groupby

bad_snake = [(432, 288), (360, 288), (360, 360), (432, 360), (432, 432), (504, 432), (504, 504), (576, 504), (576, 576), (648, 576), (648, 648), (576, 648), (504, 648), (504, 576), (432, 576), (432, 504), (360, 504), (360, 432), (288, 432), (288, 360), (216, 360), (216, 288), (144, 288), (144, 216), (216, 216), (288, 216)]
good_snake = [(144, 288), (144, 216), (216, 216), (288, 216), (360, 216), (432, 216), (504, 216), (576, 216), (576, 144), (504, 144), (432, 144), (360, 144), (288, 144), (216, 144), (144, 144), (144, 72), (216, 72), (288, 72), (360, 72), (432, 72), (504, 72), (576, 72), (648, 72), (648, 144), (648, 216), (648, 288)]

def better_estimator(snake, nb_row, nb_col):
    cell_size = 800 // nb_row
    weighted_row_score = 0
    weighted_col_score = 0

    # Normalize snake coordinates to grid cells
    snake = [(x // cell_size, y // cell_size) for x, y in snake]

    # Weighted streaks for rows
    for i in range(nb_row):
        true_false_snake_row = [0 if y != i else 1 for x, y in snake]
        row_streaks = [len(list(group)) for key, group in groupby(true_false_snake_row) if key == 1]
        # Exclude streaks of length 2 and add weighted score
        weighted_row_score += sum(streak ** 2 for streak in row_streaks if streak != 2)

    # Weighted streaks for columns
    for j in range(nb_col):
        true_false_snake_col = [0 if x != j else 1 for x, y in snake]
        col_streaks = [len(list(group)) for key, group in groupby(true_false_snake_col) if key == 1]
        # Exclude streaks of length 2 and add weighted score
        weighted_col_score += sum(streak ** 2 for streak in col_streaks if streak != 2)
    
    normalized_row_score = weighted_row_score / (nb_row ** 3)
    normalized_col_score = weighted_col_score / (nb_col ** 3)

    sum_score = normalized_row_score + normalized_col_score

    return sum_score
        
if __name__ == "__main__":
    print("Bad Snake Score :", better_estimator(bad_snake, 11, 11))
    print("Good Snake Score :", better_estimator(good_snake, 11, 11))