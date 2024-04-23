#snake_game.py
import pygame
import numpy as np
import random as rd

class SnakeGame:
    def __init__(self):
        pygame.font.init()
        self.n = 4
        self.CELL_SIZE = 80
        self.WINDOW_WIDTH = self.n * self.CELL_SIZE
        self.WINDOW_HEIGHT = self.n * self.CELL_SIZE
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.position = (self.n//2 * self.CELL_SIZE, self.n//2 * self.CELL_SIZE) # head position
        self.snake = [self.position] # body positions
        self.food = self.generate_food()
        self.steps_taken = 0
        self.visited = [] # cells visited by the snake
        self.max_steps = 100
        self.direction = 'RIGHT' # starts lookig right
        self.previous_action = [1,1] # 2 previous actions, initial set as straight - straight

    def generate_food(self):
        while True:
            food = (rd.randint(0, self.n - 1) * self.CELL_SIZE, rd.randint(0, self.n - 1) * self.CELL_SIZE)
            if food not in self.snake:
                return food

    def lost(self): # Got Outside the Map or Roll over
        return not (0 <= self.position[0] < self.WINDOW_WIDTH) or not (0 <= self.position[1] < self.WINDOW_HEIGHT) or self.position in self.snake[1:] 
    
    # Maybe go_right, go_left and go_straight can be refactored
    def go_right(self): 
        if self.direction == 'UP':
            self.position = (self.position[0] + self.CELL_SIZE, self.position[1])
            self.direction = 'RIGHT'
        elif self.direction == 'DOWN':
            self.position = (self.position[0] - self.CELL_SIZE, self.position[1])
            self.direction = 'LEFT'
        elif self.direction == 'RIGHT':
            self.position = (self.position[0], self.position[1] + self.CELL_SIZE)
            self.direction = 'DOWN'
        elif self.direction == 'LEFT':
            self.position = (self.position[0], self.position[1] - self.CELL_SIZE)
            self.direction = 'UP'

    def go_left(self):
        if self.direction == 'UP':
            self.position = (self.position[0] - self.CELL_SIZE, self.position[1])
            self.direction = 'LEFT'
        elif self.direction == 'DOWN':
            self.position = (self.position[0] + self.CELL_SIZE, self.position[1])
            self.direction = 'RIGHT'
        elif self.direction == 'LEFT':
            self.position = (self.position[0], self.position[1] + self.CELL_SIZE)
            self.direction = 'DOWN'
        elif self.direction == 'RIGHT':
            self.position = (self.position[0], self.position[1] - self.CELL_SIZE)
            self.direction = 'UP'

    def go_straight(self):
        if self.direction == 'UP':
            self.position = (self.position[0], self.position[1] - self.CELL_SIZE)
            self.direction = 'UP'
        elif self.direction == 'LEFT':
            self.position = (self.position[0] - self.CELL_SIZE, self.position[1])
            self.direction = 'LEFT'
        elif self.direction == 'RIGHT':
            self.position = (self.position[0] + self.CELL_SIZE, self.position[1])
            self.direction = 'RIGHT'
        elif self.direction == 'DOWN':
            self.position = (self.position[0], self.position[1] + self.CELL_SIZE)
            self.direction = 'DOWN'
    
    def get_direction_index(self):
        if self.direction == 'UP':
            return 0
        elif self.direction == 'RIGHT':
            return 1
        elif self.direction == 'DOWN':
            return 2
        elif self.direction == 'LEFT':
            return 3
      
    # That one too
    def get_danger(self): # is the wall on the left - ahead - right - self on the left - self ahead - self on the right
        if self.direction == 'UP':
            left_wall = int(self.position[0] // self.CELL_SIZE == 0)
            ahead_wall = int(self.position[1] // self.CELL_SIZE == 0)
            right_wall = int(self.n - 1 - self.position[0] // self.CELL_SIZE == 0)
            left_self = int((self.position[0] - self.CELL_SIZE, self.position[1]) in self.snake)
            ahead_self = int((self.position[0], self.position[1] - self.CELL_SIZE) in self.snake)
            right_self = int((self.position[0] + self.CELL_SIZE, self.position[1]) in self.snake)
            return np.array([left_wall, ahead_wall, right_wall, left_self, ahead_self, right_self])
        
        elif self.direction == 'DOWN':
            left_wall = int(self.n - 1 - self.position[0] // self.CELL_SIZE == 0)
            ahead_wall = int(self.n - 1 - self.position[1] // self.CELL_SIZE == 0)
            right_wall = int(self.position[0] // self.CELL_SIZE == 0)
            left_self = int((self.position[0] + self.CELL_SIZE, self.position[1]) in self.snake)
            ahead_self = int((self.position[0], self.position[1] + self.CELL_SIZE) in self.snake)
            right_self = int((self.position[0] - self.CELL_SIZE, self.position[1]) in self.snake)
            return np.array([left_wall, ahead_wall, right_wall, left_self, ahead_self, right_self])
        
        elif self.direction == 'LEFT':
            left_wall = int(self.n - 1 - self.position[1] // self.CELL_SIZE == 0)
            ahead_wall = int(self.position[0] // self.CELL_SIZE == 0)
            right_wall = int(self.position[1] // self.CELL_SIZE == 0)
            left_self = int((self.position[0], self.position[1] + self.CELL_SIZE) in self.snake)
            ahead_self = int((self.position[0] - self.CELL_SIZE, self.position[1]) in self.snake)
            right_self = int((self.position[0], self.position[1] - self.CELL_SIZE) in self.snake)
            return np.array([left_wall, ahead_wall, right_wall, left_self, ahead_self, right_self])
        
        elif self.direction == 'RIGHT':
            left_wall = int(self.position[1] // self.CELL_SIZE == 0)
            ahead_wall = int(self.n - 1 - self.position[0] // self.CELL_SIZE == 0)
            right_wall = int(self.n - 1 - self.position[1] // self.CELL_SIZE == 0)
            left_self = int((self.position[0], self.position[1] - self.CELL_SIZE) in self.snake)
            ahead_self = int((self.position[0] + self.CELL_SIZE, self.position[1]) in self.snake)
            right_self = int((self.position[0], self.position[1] + self.CELL_SIZE) in self.snake)
            return np.array([left_wall, ahead_wall, right_wall, left_self, ahead_self, right_self])   
    
    def get_where_food(self): # is it on the left, is it ahead , is it on the right, is it behind
        if self.direction == 'UP':
            return np.array([int(self.food[0] // self.CELL_SIZE - self.position[0] // self.CELL_SIZE < 0), int(self.position[1] // self.CELL_SIZE - self.food[1] // self.CELL_SIZE > 0), int(self.food[0] // self.CELL_SIZE - self.position[0] // self.CELL_SIZE > 0),int(self.position[1] // self.CELL_SIZE - self.food[1] // self.CELL_SIZE < 0) ])
        elif self.direction == 'DOWN':
            return np.array([int(self.position[0] // self.CELL_SIZE - self.food[0] // self.CELL_SIZE > 0), int(self.position[1] // self.CELL_SIZE - self.food[1] // self.CELL_SIZE < 0), int(self.position[0] // self.CELL_SIZE - self.food[0] // self.CELL_SIZE < 0), int(self.position[1] // self.CELL_SIZE - self.food[1] // self.CELL_SIZE > 0)])
        elif self.direction == 'LEFT':
            return np.array([int(self.food[1] // self.CELL_SIZE - self.position[1] // self.CELL_SIZE > 0 ), int(self.position[0] // self.CELL_SIZE - self.food[0] // self.CELL_SIZE > 0), int(self.food[1] // self.CELL_SIZE - self.position[1] // self.CELL_SIZE < 0 ), int(self.position[0] // self.CELL_SIZE - self.food[0] // self.CELL_SIZE < 0) ])
        elif self.direction == 'RIGHT':
            return np.array([int(self.position[1] // self.CELL_SIZE - self.food[1] // self.CELL_SIZE < 0 ), int(self.food[0] // self.CELL_SIZE - self.position[0] // self.CELL_SIZE < 0 ), int(self.position[1] // self.CELL_SIZE - self.food[1] // self.CELL_SIZE > 0 ), int(self.food[0] // self.CELL_SIZE - self.position[0] // self.CELL_SIZE > 0 )])


    def step(self, action):
        reward=0
        
        previous_distance_from_food = np.linalg.norm(np.array(self.position) - np.array(self.food))

        if action == 0:  # Left
            self.go_left()
        elif action == 1:  # Straight
            self.go_straight()
        elif action == 2:  # Right
            self.go_right()

        self.steps_taken += 1

        current_distance_from_food = np.linalg.norm(np.array(self.position) - np.array(self.food))

        # Check if the snake has reached the food
        if self.position == self.food:
            print("food reached")
            self.snake.insert(0, self.position)
            self.food = self.generate_food()
            self.visited = []
            reward += 300 + 50*len(self.snake) # the longer the snake, the greater the reward 

        else: # the longer the snake, the greater the reward move forward
            self.snake.insert(0, self.position) 
            self.snake.pop()

        if not self.lost() and (self.position in self.visited) and current_distance_from_food >= previous_distance_from_food: # did the snake got further from food ?
            done = False
            reward -= 3
        elif not self.lost() and (self.position not in self.visited) and current_distance_from_food < previous_distance_from_food: # did the snake got closer to food ?
            self.visited.append(self.position)
            done = False
            reward += 7

        if self.previous_action[0] == action and self.previous_action[1]==action and action != 1 : # Does the snake go left several times ?
            done = False
            reward -= 30 # Prevent the snake from going around in circles

        if (self.steps_taken >= self.max_steps): # Stop the game but do not penalize
            done = True
            print('Did it reached max steps ? ', (self.steps_taken >= self.max_steps))

        # elif len(self.snake)> 15:
        #     print('#############')
        #     print('###SUCCESS###')
        #     print('#############')   
        #     done = True
        #     reward += 300

        elif self.lost():
            print("Game over")
            print('Did it roll over ? ', self.position in self.snake[1:])
            print('Did it got outside of the map?', not (0 <= self.position[0] < self.WINDOW_WIDTH) or not (0 <= self.position[1] < self.WINDOW_HEIGHT))
            done = True
            reward -= 100
        
        else:
            done = False
            reward -= 4 # Thrust the snake to rush the food
            print('Nothing happened')
        
        self.previous_action.pop(0)
        self.previous_action.append(action)

        print('reward += ', reward)
        return self.get_state(), reward, done, {}

    def reset(self,max_steps,N):
        self.n = N
        self.WINDOW_WIDTH = self.n * self.CELL_SIZE
        self.WINDOW_HEIGHT = self.n * self.CELL_SIZE
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        self.position = ((self.n//2)* self.CELL_SIZE, self.n//2*self.CELL_SIZE)
        self.snake = [self.position]
        self.food = self.generate_food()
        self.steps_taken = 0
        self.visited = []
        self.max_steps = max_steps
        self.direction = 'RIGHT'
        self.previous_action = [1,1]

    def get_state(self):
        distance_to_walls = self.get_danger() # is the wall on the left - ahead - right - self on the left - self ahead - self on the right
        distance_to_food = self.get_where_food() # is it on the left, is it ahead, is it on the right, is it behind
        snake_direction = np.array([self.get_direction_index()]) # 2 Last actions
        full_state = np.concatenate((distance_to_walls,distance_to_food,snake_direction))
        return full_state

    def render(self,rendering,reward,clock):
        if rendering :
            self.screen.fill((245, 245, 220)) # cream

            # Draw the snake segments
            for i, segment in enumerate(self.snake):
                segment_color = (60, 60, 60) if i == 0 else (20, 20, 20)  # Head is darker grey
                pygame.draw.rect(self.screen, segment_color, (segment[0], segment[1], self.CELL_SIZE, self.CELL_SIZE))
                pygame.draw.rect(self.screen, (0, 0, 0), (segment[0], segment[1], self.CELL_SIZE, self.CELL_SIZE), 2)  # Black outline

            # Draw the food
            pygame.draw.circle(self.screen, (255, 0, 0), (self.food[0] + self.CELL_SIZE // 2, self.food[1] + self.CELL_SIZE // 2), self.CELL_SIZE // 3)

            # Draw the text in the top right corner
            score_font = pygame.font.Font(pygame.font.get_default_font(), 14)
            score_text = score_font.render(f"Current reward : {reward} Steps : {self.steps_taken} Score: {len(self.snake) - 1}", True, (0, 0, 0))
            score_rect = score_text.get_rect()
            score_rect.topright = (self.WINDOW_WIDTH - 10, 10)
            self.screen.blit(score_text, score_rect)

            pygame.display.flip()
            self.clock.tick(clock)  # Increase the frame rate for smoother rendering

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

"""
If you want to play the snake game yourself, set True.
It's not the real snake game, you can go left, straight and right. It's like stop motion.
Press s to get the state.
"""

if False :
    # Initialize Pygame
    pygame.init()

    # Create the game environment
    game = SnakeGame()

    running = True
    total_reward=0
    make_step=False
    done = False

    # Main loop
    while running:
        
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
                elif event.key == pygame.K_s:
                    print('State :', game.get_state())
                    print('snake : ',game.snake)
            
            if make_step :
                next_state, reward, done, _ = game.step(action)
                total_reward+=reward
                make_step=False
            
            game.render(True,total_reward,12)

            if done:
                game.reset(100,10)
                total_reward=0
                done = False