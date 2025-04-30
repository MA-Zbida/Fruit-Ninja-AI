import torch
import keyboard
import cv2
import mss
import time
import numpy as np
from VideoRecorder import VideoRecorder
from pynput.mouse import Button, Controller
from ultralytics import YOLO
from ScreenCapture import capture_fullscreen
from Game import Game
from pynput.mouse import Button, Controller


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
    model_path = "Model YOLOv8n/fruit_ninja_modelv8.pt" # you can use "Model YOLOv5n/fruit_ninja_modelv5.pt"
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

main()