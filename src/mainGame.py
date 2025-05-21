import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import mss
import pygetwindow as gw
import keyboard
from pynput.mouse import Button, Controller
from ultralytics import YOLO
import os
from datetime import datetime

class Fruit:
    def __init__(self, xywhn: torch.Tensor):
        self.xywhn = xywhn
        self.x = float(xywhn[0])
        self.y = float(xywhn[1])
        self.width = float(xywhn[2])
        self.height = float(xywhn[3])

class Game:
    def __init__(self):
        self.bombs = None
        self.fruits = []
        self.maze_size = 300
        self.size_multiplier = 1.1
        self.bomb_size = 0.1529 * self.size_multiplier
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

def _distance(start, end):
    return abs(start[0] - end[0]) + abs(start[1] - end[1])
        
def modified_astar(start: tuple, end: tuple, bombs: torch.Tensor, point_inside_bomb: callable, map_size: int, interval: int=1):
    # Find the shortest path around the bombs with improved pathing for fruits near bombs
    # Consider 8 directions instead of 4 to allow for more flexible paths
    directions = [(0, 1), (0, -1), (-1, 0), (1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: _distance(start, end)}
    open_heap = []
    heapq.heappush(open_heap, (fscore[start], start))
    
    # Allow for a more direct path toward the end by weighting heuristic
    heuristic_weight = 1.2
    
    while open_heap:
        # Get the point with the lowest fscore
        current = heapq.heappop(open_heap)[1]
        # If the current point is the end point, reconstruct the path and return it
        if current == end:
            path = deque()
            i = interval
            while current in came_from:
                if i == interval:
                    path.appendleft(current)
                    i = 0
                current = came_from[current]
                i += 1
            # Add the last point on the end fruit
            if start in came_from:
                path.appendleft(start)
            return list(path)
        
        # Add the current point to the close set
        close_set.add(current)
        # Check the neighbors of the current point
        for dx, dy in directions:
            # Get the neighbor point
            neighbor = (current[0] + dx, current[1] + dy)
            # If the neighbor is in the close set, outside the screen, or inside a bomb, skip it
            if (neighbor in close_set or 
                neighbor[0] < 0 or neighbor[0] >= map_size or 
                neighbor[1] < 0 or neighbor[1] >= map_size or 
                point_inside_bomb(neighbor[0], neighbor[1], True)):
                close_set.add(neighbor)
                continue
                
            # Give diagonal moves a slightly higher cost (sqrt(2) ≈ 1.414)
            move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
            
            # Calculate the gscore of the neighbor
            tentative_gscore = gscore[current] + move_cost
            # If the neighbor is not in the open set or the gscore is lower than the current gscore of the neighbor
            if (neighbor not in gscore or tentative_gscore < gscore[neighbor]):
                # Update the gscore, fscore, and add the neighbor to the open set
                came_from[neighbor] = current
                gscore[neighbor] = tentative_gscore
                # Weight the heuristic to favor more direct paths
                fscore[neighbor] = gscore[neighbor] + heuristic_weight * _distance(neighbor, end)
                heapq.heappush(open_heap, (fscore[neighbor], neighbor))
    # If there is no path, return an empty list
    return []

        
def capture_fullscreen(new_size=(640, 640)):
    # Capture the entire screen
    with mss.mss() as sct:
        # Get the primary monitor
        monitor = sct.monitors[1]  # monitor 1 is the primary monitor
        
        # Capture the screen
        screenshot = sct.grab(monitor)
        
        # Convert to numpy array
        frame = np.array(screenshot)
        
        # Convert BGRA to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Resize if needed
        if new_size != (frame.shape[1], frame.shape[0]):
            frame = cv2.resize(frame, new_size)
            
        return frame, monitor["left"], monitor["top"], monitor["width"], monitor["height"]

def precise_sleep(duration):
    # Sleep for a precise duration as time.sleep is not accurate at very small durations
    start_time = time.time()
    while time.time() < start_time + duration:
        pass

def move(to_position, delay=1):
    # Move the mouse to the given position and wait for the given duration
    mouse.position = to_position
    precise_sleep(delay)

def mouse_circle(radius, step_duration=1):
    # Move the mouse in a circle with the center at the current position and the given radius
    start_position = mouse.position
    # Number of points to have in the circle
    steps = 25
    # Check if the angles for the given radius are cached
    angles = cached_angle.get(radius, None)
    # If the angles are not cached, calculate them
    if (angles is None):
        angles = np.linspace(0, 2 * np.pi, steps)
        cos = np.cos(angles) * radius
        sin = np.sin(angles) * radius
        cached_angle[radius] = (cos, sin)
    # If the angles are cached, get the cosine and sine values
    else:
        cos, sin = angles
    # Calculate the new positions for the mouse using the cosine and sine values
    x_positions = start_position[0] + cos
    y_positions = start_position[1] + sin
    # Move the mouse to the new positions
    for i in range(steps):
        new_x = x_positions[i]
        new_y = y_positions[i]
        mouse.position = (new_x, new_y)
        precise_sleep(step_duration)

def cut_fruit(path: np.ndarray, app_x, app_y, app_width, app_height):
    # Cut the fruit along the path
    if len(path) == 0:
        return
    # Convert the path to the screen coordinates
    path[:, 0] = (path[:, 0] * app_width + app_x).astype(int)
    path[:, 1] = (path[:, 1] * app_height + app_y).astype(int)
    # Get the first point from the path
    first_point = path[0]
    path = path[1:]
    # Move the mouse to the first point
    mouse.position = tuple(first_point)
    # Start slicing the fruit
    mouse.press(Button.left)
    # Move the mouse in a circle to slice the first fruit
    mouse_circle(25, 0.0000002)
    # Move the mouse to the rest of the points in the path
    time_per_point = 0.0000002
    for point in path:
        # Move the mouse to the point
        move(tuple(point), time_per_point)
    # Move the mouse in a circle to slice the last fruit
    mouse_circle(25, 0.0000002)
    mouse.release(Button.left)

class VideoRecorder:
    def __init__(self, output_dir="recordings", fps=120):
        self.output_dir = output_dir
        self.fps = fps
        self.recording = False
        self.video_writer = None
        self.frame_count = 0
        self.start_time = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def start_recording(self, frame_shape):
        if self.recording:
            return
            
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"gameplay_{timestamp}.mp4")
        
        # Initialize video writer
        height, width = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
        self.video_writer = cv2.VideoWriter(filename, fourcc, self.fps, (width, height))
        self.recording = True
        self.frame_count = 0
        self.start_time = time.time()
        print(f"Recording started: {filename}")
        
    def add_frame(self, frame):
        if not self.recording:
            return
            
        # Assume frame is already in BGR format - don't convert incorrectly
        self.video_writer.write(frame)
        self.frame_count += 1
        
    def stop_recording(self):
        if not self.recording:
            return
            
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
            
        duration = time.time() - self.start_time
        print(f"Recording stopped. {self.frame_count} frames recorded over {duration:.2f} seconds.")
        print(f"Effective FPS: {self.frame_count / duration:.2f}")

def main():
    # Global variables
    global mouse, sct, cached_app_position, cached_angle
    
    # Initialize global variables
    mouse = Controller()
    cached_app_position = []
    cached_angle = {}
    sct = mss.mss()
    
    # Select device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Set the GPU to use (0, 1, 2, or 3)
        print("Using GPU")  # Hopefully it detects for faster inference
    else:
        print("Using CPU")
        
    # Initialize model with higher confidence threshold
    model_path = "bestyolov8n.pt"
    model = YOLO(model_path).to(device)
    
    # Warmup the model to improve initial performance
    print("Warming up model...")
    dummy_input = torch.zeros((1, 3, 640, 640))
    model(dummy_input, device=device, half=True, imgsz=(640, 640))
    
    # Initialize game with higher confidence settings
    game = Game()
    game.max_empty_frames = 3  # More aggressive at responding to empty frames
    game.bomb_buffer = 0.001    # Allow cutting closer to bombs
    
    # Initialize video recorder
    recorder = VideoRecorder(fps=60)
    is_recording = False
    
    # Initialize processing parameters
    image_size = (640, 640)
    
    # Higher confidence and IOU thresholds for more reliable detection
    conf_threshold = 0.6
    iou_threshold = 0.3  
    
    print("Starting Fruit Ninja Bot")
    print("Press 'r' to start/stop recording")
    print("Press 'q' to quit")
    
    try:
        while True:
            # Check if the 'q' key has been pressed to quit
            if keyboard.is_pressed('q'):
                break
                
            # Check if 'r' key has been pressed to toggle recording
            if keyboard.is_pressed('r'):
                is_recording = not is_recording
                if is_recording:
                    # Start recording on next frame (we need the frame shape)
                    pass
                else:
                    recorder.stop_recording()
                # Wait for key release to prevent multiple toggles
                while keyboard.is_pressed('r'):
                    pass
                
            # Capture the screenshot of the desired region
            screenshot, app_x, app_y, app_width, app_height = capture_fullscreen(new_size=image_size)
            
            # If the screenshot is None it means there is no new image, so skip the current iteration
            if screenshot is None:
                continue
                
            # Start recording if needed (now we have the frame shape)
            if is_recording and not recorder.recording:
                recorder.start_recording(screenshot.shape)
                
            # Predict the screenshot using the YOLO model with higher confidence
            predictions = model(screenshot, device=device, verbose=False, 
                               iou=iou_threshold, conf=conf_threshold, 
                               int8=True, imgsz=image_size)
                               
            # Get prediction classes and bounding boxes
            boxes = predictions[0].boxes
            classes = boxes.cls
            bounding_boxes = boxes.xywhn
            
            # Get an image with the bounding boxes and classes
            frame = predictions[0].plot()
            
            # Count bombs and fruits (fix the .item() issue)
            num_bombs = int(torch.sum(classes == 0)) if isinstance(classes, torch.Tensor) else sum(c == 0 for c in classes)
            num_fruits = int(torch.sum(classes == 1)) if isinstance(classes, torch.Tensor) else sum(c == 1 for c in classes)
            
            # Show the frame with detections (assuming frame is already in BGR format from YOLO's plot())
            cv2.imshow('Fruit Ninja Bot View', frame)
            
            # Add frame to recording if active
            if recorder.recording:
                recorder.add_frame(frame)
                
            # Update the game state
            game.update(classes, bounding_boxes)
            
            # Only act if the model is confident
            if game.is_confident():
                # Get the path between the fruits that the fruit ninja should take
                cut_path = np.array(game.get_fruit_path())
                cut_fruit(cut_path, app_x, app_y, app_width, app_height)
            
            # Wait for a short duration before capturing the next screenshot
            if cv2.waitKey(1) == ord('q'):  # waitKey argument is in milliseconds
                break
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Clean up
        if recorder.recording:
            recorder.stop_recording()
        cv2.destroyAllWindows()
        sct.close()  # Close mss when done capturing
        print("Fruit Ninja Bot stopped")

if __name__ == '__main__':
    import heapq
    from collections import deque
    main()