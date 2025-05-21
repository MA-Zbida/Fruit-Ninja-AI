import mss
import cv2
import numpy as np

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