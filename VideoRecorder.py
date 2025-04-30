import os
import time
import cv2
from datetime import datetime

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