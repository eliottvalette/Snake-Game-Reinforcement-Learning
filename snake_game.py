#snake_game.py
import pygame
import numpy as np
import random as rd
import time
from hard_code_pattern_11 import pattern_11

GLOBAL_N = 11
INITIAL_WALLS = 0

# CONSTANTS
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
STRAIGHT = 0
LEFT_TURN = -1
RIGHT_TURN = 1

class SnakeGame:
    def __init__(self):
        pygame.font.init()
        self.n = GLOBAL_N
        self.CELL_SIZE = 800 // GLOBAL_N
        self.WINDOW_WIDTH = self.CELL_SIZE * self.n
        self.WINDOW_HEIGHT = self.CELL_SIZE * self.n
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.position = (rd.randint(1, GLOBAL_N-1) * self.CELL_SIZE, rd.randint(1, GLOBAL_N-1) * self.CELL_SIZE) # head position
        self.snake = self.create_initial_snake(length = 1)
        self.walls = [] 
        self.food = self.generate_food()
        self.steps_taken = 0
        self.visited = [] # cells visited by the snake
        self.max_steps = 200
        self.direction = RIGHT # starts lookig right
        self.previous_action = [1,1] # 2 previous actions, initial set as straight - straight
        self.epsilon = 1
        self.steps_after_food = 0
        self.viable_pattern = pattern_11
        for _ in range (INITIAL_WALLS):
            self.add_wall()

    def reset(self,max_steps, N, length = 1):
        self.n = N
        self.CELL_SIZE = 800 // GLOBAL_N
        self.WINDOW_WIDTH = self.CELL_SIZE * self.n
        self.WINDOW_HEIGHT = self.CELL_SIZE * self.n
        self.position = (rd.randint(1, GLOBAL_N-1) * self.CELL_SIZE, rd.randint(1, GLOBAL_N-1) * self.CELL_SIZE) # head position
        self.snake = self.create_initial_snake(length)
        self.walls = []
        self.food = self.generate_food()
        self.steps_taken = 0
        self.visited = []
        self.max_steps = max_steps
        self.direction = RIGHT
        self.previous_action = [1,1]
        self.steps_after_food = 0
        for _ in range (INITIAL_WALLS):
            self.add_wall()

    def create_initial_snake(self, length=1):
        initial_snake = [self.position]
        for i in range(1, length):
            initial_snake.append((self.position[0] - i * self.CELL_SIZE, self.position[1]))
        return initial_snake

    def generate_food(self):
        while True:
            food = (rd.randint(0, self.n - 1) * self.CELL_SIZE, rd.randint(0, self.n - 1) * self.CELL_SIZE)
            if food not in self.snake and food not in self.walls:
                return food

    def lost(self): # Got Outside the Map or Roll over
        out_in_width = not (0 <= self.position[0] < self.WINDOW_WIDTH)
        out_in_height = not (0 <= self.position[1] < self.WINDOW_HEIGHT)
        rolled_over = self.position in self.snake[1:] 
        banged_a_wall = self.position in self.walls
        return out_in_width or out_in_height or rolled_over or banged_a_wall
    

    def add_wall(self, max_attempts=100):
        attempts = 0
        while attempts < max_attempts:
            (x_wall, y_wall) = rd.choice(self.viable_pattern)
            wall = (x_wall * self.CELL_SIZE, y_wall * self.CELL_SIZE)
            if wall not in self.snake and wall != self.food and wall not in self.walls:
                self.walls.append(wall)
                break
            attempts += 1

    def move(self, action):
        direction_vectors = {
            UP: (0, -1),
            RIGHT: (1, 0),
            DOWN: (0, 1),
            LEFT: (-1, 0)
        }

        # Convert actions to changes in direction
        action_to_turn = {
            'RIGHT': RIGHT_TURN,
            'LEFT': LEFT_TURN,
            'STRAIGHT': STRAIGHT
        }

        self.direction = (self.direction + action_to_turn[action]) % 4
        movement = direction_vectors[self.direction]

        self.position = (
            self.position[0] + movement[0] * self.CELL_SIZE,
            self.position[1] + movement[1] * self.CELL_SIZE
        )

    def get_direction_array(self):
        array = np.ones(4) * 0.01
        direction_idx = self.direction
        array[direction_idx] = 1
        return array

    def get_danger(self):
        # Define relative position changes for left, ahead, and right based on the current direction
        danger_offsets = {
            UP: [(-1, 0), (0, -1), (1, 0)],    # Left, ahead, right relative to UP
            RIGHT: [(0, -1), (1, 0), (0, 1)],  # Left, ahead, right relative to RIGHT
            DOWN: [(1, 0), (0, 1), (-1, 0)],   # Left, ahead, right relative to DOWN
            LEFT: [(0, 1), (-1, 0), (0, -1)]   # Left, ahead, right relative to LEFT
        }

        # Get the offsets relative to the current direction
        relative_positions = danger_offsets[self.direction]
        
        danger = []
        
        # Check each of the three positions: left, ahead, right
        for dx, dy in relative_positions:
            new_x = self.position[0] + dx * self.CELL_SIZE
            new_y = self.position[1] + dy * self.CELL_SIZE
            
            # Check if there's danger from the wall, snake body, or being out of bounds
            wall_danger = not (0 <= new_x < self.WINDOW_WIDTH and 0 <= new_y < self.WINDOW_HEIGHT)
            snake_body_danger = (new_x, new_y) in self.snake[1:]  # Exclude head from snake body check
            
            # Mark danger if the new position is out of bounds, a wall, or the snake's body
            danger.append(int(wall_danger))
            danger.append(int(snake_body_danger))          
        return np.array(danger)

    def get_where_food(self):
        # Calculate the relative position of the food with respect to the head of the snake
        relative_food_x = self.food[0] - self.position[0]
        relative_food_y = self.food[1] - self.position[1]

        # Directional offsets to check for food relative to the snake's current direction
        direction_offsets = {
            UP: [(0, -1), (-1, 0), (1, 0)],    # Ahead, left, right when facing UP
            RIGHT: [(1, 0), (0, -1), (0, 1)],  # Ahead, left, right when facing RIGHT
            DOWN: [(0, 1), (1, 0), (-1, 0)],   # Ahead, left, right when facing DOWN
            LEFT: [(-1, 0), (0, 1), (0, -1)]   # Ahead, left, right when facing LEFT
        }

        # Get the directional offsets based on the current direction of the snake
        relative_positions = direction_offsets[self.direction]

        where_food = []

        # Check if the food is ahead, to the left, or to the right
        for dx, dy in relative_positions:
            food_ahead = (relative_food_x * dx >= 0) and (relative_food_y * dy >= 0)
            where_food.append(int(food_ahead))

        return np.array(where_food)

    def shrink_matrix(self, matrix):
        shrink = (self.n - 11) // 2
        if shrink != 0 :
            matrix = matrix[shrink : -shrink, shrink : -shrink]
        return matrix

    def get_board_matrix(self):
        board = np.ones((self.n, self.n))*0.01

        if not self.lost():

            for i, j in self.snake[1:]:
                resized_i, resized_j = (i // self.CELL_SIZE, j // self.CELL_SIZE) 
                board[resized_j, resized_i] = -1
            
            for i, j in self.walls:
                resized_i, resized_j = (i // self.CELL_SIZE, j // self.CELL_SIZE) 
                board[resized_j, resized_i] = -1
            
            board[self.position[1] // self.CELL_SIZE , self.position[0] // self.CELL_SIZE] = 1

            board[self.food[1] // self.CELL_SIZE , self.food[0] // self.CELL_SIZE] = 1

        center_x = center_y = self.n // 2
        head_x, head_y = self.position[1] // self.CELL_SIZE, self.position[0]// self.CELL_SIZE
        distance_to_center_x, distance_to_center_y = head_x - center_x, head_y - center_y

        shifted_board = np.ones((self.n, self.n))*(-1)
        for i in range(self.n):
            for j in range(self.n):
                new_i, new_j = i - distance_to_center_x, j - distance_to_center_y
                if 0 <= new_i < self.n and 0 <= new_j < self.n:
                    shifted_board[new_i, new_j] = board[i, j]

        number_of_rotation = self.direction
        shifted_board = np.rot90(shifted_board, number_of_rotation)

        shrink_matrix = self.shrink_matrix(shifted_board)

        return shrink_matrix.tolist(), shrink_matrix

    def calculate_wall_reward(self):
        x, y = self.position
        distance_left = x // self.CELL_SIZE  # Cells to the left wall
        distance_right = (self.WINDOW_WIDTH - x - self.CELL_SIZE) // self.CELL_SIZE  # Cells to the right wall
        distance_top = y // self.CELL_SIZE  # Cells to the top wall
        distance_bottom = (self.WINDOW_HEIGHT - y - self.CELL_SIZE) // self.CELL_SIZE  # Cells to the bottom wall

        min_distance = min(distance_left, distance_right, distance_top, distance_bottom)

        if min_distance == 0:
            reward = 10 
        elif min_distance == 1:
            reward = 5
        elif min_distance == 2:
            reward = 2
        else:
            reward = 0 

        return reward


    def step(self, action):
        reward = +0.5  # Base reward for taking a step

        previous_distance_from_food = np.linalg.norm(np.array(self.position) - np.array(self.food))

        # Move the snake based on the action
        if action == 0:  # Left
            self.move('LEFT')
        elif action == 1:  # Straight
            self.move('STRAIGHT')
        elif action == 2:  # Right
            self.move('RIGHT')

        self.steps_taken += 1

        current_distance_from_food = np.linalg.norm(np.array(self.position) - np.array(self.food))

        # Distance-based reward: Encourage moving towards food
        if current_distance_from_food < previous_distance_from_food and action == 1:
            reward += 3.5  # Reward for moving closer to the food
        elif current_distance_from_food < previous_distance_from_food and action != 1:
            reward += 1
        elif current_distance_from_food > previous_distance_from_food:
            reward -= 0.5  # Penalty for moving away from the food

        # Check if the snake has reached the food
        if self.position == self.food:
            self.snake.insert(0, self.position)  # Extend the snake
            self.food = self.generate_food()  # Generate new food
            reward += 150  # Reward for eating food
            self.steps_after_food = 0  # Reset step count after food
            self.visited = []  # Reset visited positions after eating food
        else:
            self.snake.insert(0, self.position)
            self.snake.pop()  # Remove the tail if no food was eaten

        # Check if the snake is done
        if self.lost():
            reward -= 600  # Penalty for losing
            done = True
        elif self.steps_taken >= self.max_steps:
            reward -= 300 
            done = True
            # print('Reached max steps')
        else:
            done = False

        self.steps_after_food += 1
        self.previous_action.pop(0)
        self.previous_action.append(action)

        return self.get_state(), reward, done, {}

    def get_state(self):
        board_matrix = np.array(self.get_board_matrix()[0]).flatten() # 11x11 matrix (121)
        distance_to_walls = self.get_danger() # is the wall on the left - ahead - right same for the snake itself (6)
        distance_to_food = self.get_where_food() # is it on the ahead, is it right, is it on the left(3)
        snake_direction = self.get_direction_array() # direction (4)
        full_state = np.concatenate((board_matrix, distance_to_walls, distance_to_food, snake_direction))
        full_state = np.where(full_state == 0, 0.01, full_state)  # Replace 0 with 0.01
        
        return full_state

    def render(self, rendering, reward, clock):
        if rendering:
            # Fill background with a gradient effect
            self.screen.fill((240, 240, 240))  # Light gray base
            
            # Draw grid lines for better visibility
            for i in range(self.n + 1):
                pygame.draw.line(self.screen, (200, 200, 200), 
                            (i * self.CELL_SIZE, 0), 
                            (i * self.CELL_SIZE, self.WINDOW_HEIGHT))
                pygame.draw.line(self.screen, (200, 200, 200), 
                            (0, i * self.CELL_SIZE), 
                            (self.WINDOW_WIDTH, i * self.CELL_SIZE))

            # Draw snake body with gradient effect
            for i, segment in enumerate(self.snake):
                # Create gradient from tail to head
                intensity = min(255, 100 + (155 * i / len(self.snake)))
                body_color = (0, intensity, 0)
                
                # Draw rounded rectangle for each segment
                rect = pygame.Rect(segment[0], segment[1], self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, body_color, rect, border_radius=self.CELL_SIZE // 4)
                pygame.draw.rect(self.screen, (0, 100, 0), rect, 2, border_radius=self.CELL_SIZE // 4)
                
                # Add eyes to head
                if i == 0:
                    eye_radius = self.CELL_SIZE // 6
                    eye_offset = self.CELL_SIZE // 4
                    
                    # Determine eye positions based on direction
                    if self.direction == RIGHT:
                        left_eye = (segment[0] + self.CELL_SIZE - eye_offset, segment[1] + eye_offset)
                        right_eye = (segment[0] + self.CELL_SIZE - eye_offset, segment[1] + self.CELL_SIZE - eye_offset)
                    elif self.direction == LEFT:
                        left_eye = (segment[0] + eye_offset, segment[1] + eye_offset)
                        right_eye = (segment[0] + eye_offset, segment[1] + self.CELL_SIZE - eye_offset)
                    elif self.direction == UP:
                        left_eye = (segment[0] + eye_offset, segment[1] + eye_offset)
                        right_eye = (segment[0] + self.CELL_SIZE - eye_offset, segment[1] + eye_offset)
                    else:  # DOWN
                        left_eye = (segment[0] + eye_offset, segment[1] + self.CELL_SIZE - eye_offset)
                        right_eye = (segment[0] + self.CELL_SIZE - eye_offset, segment[1] + self.CELL_SIZE - eye_offset)
                    
                    pygame.draw.circle(self.screen, (255, 255, 255), left_eye, eye_radius)
                    pygame.draw.circle(self.screen, (255, 255, 255), right_eye, eye_radius)
                    pygame.draw.circle(self.screen, (0, 0, 0), left_eye, eye_radius // 2)
                    pygame.draw.circle(self.screen, (0, 0, 0), right_eye, eye_radius // 2)

            # Draw walls with texture
            for wall in self.walls:
                x, y = wall
                wall_rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                
                # Draw main wall block
                pygame.draw.rect(self.screen, (100, 100, 100), wall_rect)
                pygame.draw.rect(self.screen, (50, 50, 50), wall_rect, 2)
                
                # Add brick pattern
                brick_offset = self.CELL_SIZE // 3
                pygame.draw.line(self.screen, (50, 50, 50), 
                            (x, y + brick_offset), 
                            (x + self.CELL_SIZE, y + brick_offset))
                pygame.draw.line(self.screen, (50, 50, 50), 
                            (x + brick_offset, y), 
                            (x + brick_offset, y + self.CELL_SIZE))

            # Draw food as an apple
            apple_color = (220, 20, 20)  # Bright red
            center_x = self.food[0] + self.CELL_SIZE // 2
            center_y = self.food[1] + self.CELL_SIZE // 2
            radius = self.CELL_SIZE // 2.5
            
            # Draw main apple body
            pygame.draw.circle(self.screen, apple_color, (center_x, center_y), radius)
            
            # Add stem and leaf
            stem_start = (center_x, center_y - radius)
            stem_end = (center_x, center_y - radius - self.CELL_SIZE // 6)
            pygame.draw.line(self.screen, (101, 67, 33), stem_start, stem_end, 2)
            
            # Draw leaf
            leaf_points = [
                (center_x, stem_end[1]),
                (center_x + self.CELL_SIZE // 6, stem_end[1] + self.CELL_SIZE // 8),
                (center_x, stem_end[1] + self.CELL_SIZE // 4)
            ]
            pygame.draw.polygon(self.screen, (34, 139, 34), leaf_points)

            # Draw score and info panel with better styling
            panel_height = 30
            pygame.draw.rect(self.screen, (50, 50, 50), 
                            (0, 0, self.WINDOW_WIDTH, panel_height))
            
            # Create dynamic font
            score_font = pygame.font.Font(pygame.font.get_default_font(), 16)
            
            # Format text with better spacing
            score_text = score_font.render(
                f"Score: {len(self.snake) - 1} | Reward: {reward:.1f} | Random: {self.epsilon*100:.1f}%", 
                True, (255, 255, 255))
            score_rect = score_text.get_rect()
            score_rect.midtop = (self.WINDOW_WIDTH // 2, 5)
            self.screen.blit(score_text, score_rect)

            pygame.display.flip()
            self.clock.tick(clock)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

"""
If you want to play the snake game yourself, set True.
Press s to get the state.

Don't forget to reset it as False, otherwise the training will fail
"""

if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()

    # Create the game environment
    game = SnakeGame()

    run = True
    total_reward=0
    make_step=False
    done = False

    # Main loop
    while run:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action=0
                    make_step=True
                elif event.key == pygame.K_UP:
                    action=1
                    make_step=True
                elif event.key == pygame.K_RIGHT:
                    action=2
                    make_step=True
                elif event.key == pygame.K_w:
                    game.add_wall()
                elif event.key == pygame.K_s:
                    print('State :', game.get_state())
                    print('Board :\n', game.get_board_matrix()[1])
                    print('Indicators :\n', game.get_state()[121:])
                    print('Snake : ',game.snake)

                    print('Food Direction : ', game.get_where_food())
            
            if make_step :
                next_state, reward, done, _ = game.step(action)
                total_reward+=reward
                make_step=False
            
            game.render(True, total_reward, 12)

            if done:
                game.reset(max_steps = 1000, N = GLOBAL_N, length= 6)
                total_reward=0
                done = False