import numpy as np
from Fruit import Fruit
import torch
from Astar import modified_astar

class Game:
    def __init__(self):
        self.bombs = None
        self.fruits = []
        self.maze_size = 300
        self.size_multiplier = 1.1
        self.bomb_size = 0.155 * self.size_multiplier
        self.min_fruit_height = 0.9
        self.min_start_slice_height = 0.6
        # Confidence tracking
        self.consecutive_empty_frames = 0
        self.max_empty_frames = 5
        # Bomb avoidance buffer
        self.bomb_buffer = 0.001  # Small buffer to allow closer cuts

    def get_bomb_array(self, xywhn: np.ndarray):
        # Create a tensor to store the bomb data. Store the x, y, top left x, top left y, bottom right x, bottom right y, and
        # scaled top left and bottom right x and y values
        xywhn = xywhn.cpu().numpy()
        bomb_array = np.zeros((10, 1))
        bomb_array[0:2, 0] = xywhn[0:2]
        bomb_array[2:4, 0] = xywhn[0:2] - self.bomb_size
        bomb_array[4:6, 0] = xywhn[0:2] + self.bomb_size
        # Clamp the corners to be inside the screen
        bomb_array[2:6, 0] = np.clip(bomb_array[2:6, 0], 0, 1)
        # Scale the corners to the maze size
        bomb_array[6:10, 0] = bomb_array[2:6, 0] * self.maze_size
        return bomb_array

    def add_bomb(self, bomb: np.ndarray):
        if self.bombs is None:
            self.bombs = bomb
        else:
            self.bombs = np.concatenate((self.bombs, bomb), axis=1)

    def add_fruit(self, fruit: Fruit):
        self.fruits.append(fruit)

    def clear_bombs(self):
        self.bombs = None

    def clear_fruits(self):
        self.fruits = []

    def clear_all(self):
        self.clear_bombs()
        self.clear_fruits()

    def update(self, cls: torch.Tensor, xywhn: torch.Tensor):
        # Update the fruits and bombs
        # cls is a tensor of classes where 1 is a fruit and 0 is a bomb
        # Clear the fruits and bombs
        self.clear_all()

        # Add the fruits and bombs
        bombs = xywhn[cls == 0]
        fruits = xywhn[cls == 1]
        
        # Track empty frames for confidence measurement
        if len(bombs) == 0 and len(fruits) == 0:
            self.consecutive_empty_frames += 1
        else:
            self.consecutive_empty_frames = 0
            
        for i in range(bombs.shape[0]):
            self.add_bomb(self.get_bomb_array(bombs[i]))
        # Add the fruits that are not inside a bomb and are above the minimum fruit height
        for i in range(fruits.shape[0]):
            if ((float(fruits[i][1]) < self.min_fruit_height) and (not self.point_inside_bomb(float(fruits[i][0]), float(fruits[i][1])))):
                self.add_fruit(Fruit(fruits[i]))

        # Sort the fruits so that they are in order from top left to bottom right
        self.sort_fruits()
        
    def sort_fruits(self):
        self.fruits.sort(key=lambda fruit: fruit.x)
        # Sort the fruits by their y coordinate
        self.fruits.sort(key=lambda fruit: fruit.y)
    
    def path_between_fruits(self, start: Fruit, end: Fruit):
        start_point = (int(start.x * self.maze_size), int(start.y * self.maze_size))
        end_point = (int(end.x * self.maze_size), int(end.y * self.maze_size))
        path = modified_astar(start_point, end_point, self.bombs, self.point_inside_bomb, self.maze_size, interval=10)
        return path

    def get_fruit_path(self):
        if len(self.fruits) < 1:
            return np.array([])

        if self.fruits[0].y > self.min_start_slice_height and self.fruits[0].width < 0.14:
            return np.array([])
        # If there is only 1 fruit, return a list with the position of the fruit
        if len(self.fruits) == 1:
            return np.array([(self.fruits[0].x, self.fruits[0].y)])
        
        # If there are 2 or more fruits, find the paths between the fruits
        path = []
        for i in range(len(self.fruits) - 1):
            path += self.path_between_fruits(self.fruits[i], self.fruits[i + 1])

        # Normalize the path
        return np.array(path) / self.maze_size

    def point_inside_bomb(self, x: float, y: float, rescale: bool = False):
        # Check if the point is inside any of the bombs with a buffer to allow closer paths
        if self.bombs is None:
            return False
        
        # Get the corner coordinates of the bombs
        if rescale:
            corner_coords = self.bombs[6:10, :]
            # Scale the buffer for pixel coordinates
            buffer = self.bomb_buffer * self.maze_size
        else:
            corner_coords = self.bombs[2:6, :]
            buffer = self.bomb_buffer
        
        # Check if the point is outside all the bombs (with a reduced buffer to allow closer cuts)
        outside = (x < corner_coords[0, :] - buffer) | (x > corner_coords[2, :] + buffer) | \
                  (y < corner_coords[1, :] - buffer) | (y > corner_coords[3, :] + buffer)
        return ~outside.all()
        
    def is_confident(self):
        # Returns True if the model has been consistently finding objects
        return self.consecutive_empty_frames < self.max_empty_frames