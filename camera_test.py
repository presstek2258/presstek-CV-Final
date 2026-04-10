# SCRIPT WRITTEN BY GEMINI 3 PRO
import cv2
import os
import time

# --- Configuration ---
PI_IP = "..."  
STREAM_URL = f"http://{PI_IP}:8080/?action=stream"

# Settings for the burst test
MAX_FRAMES_TO_SAVE = 30 
OUTPUT_DIR = "camera_test_frames"

def test_camera_stream():
    # 1. Create a folder to store the images
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # 2. Connect to the stream
    print(f"Connecting to {STREAM_URL}...")
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print("Failed to open stream. Make sure MasterPi.py is running on the Pi!")
        return

    print(f"Connection successful! Snapping {MAX_FRAMES_TO_SAVE} frames...")
    
    frame_count = 0
    start_time = time.time()

    # 3. The Capture Loop
    while frame_count < MAX_FRAMES_TO_SAVE:
        ret, frame = cap.read()
        
        if not ret:
            print(f"Dropped frame {frame_count}, trying again...")
            continue

        # Save the frame to the folder (e.g., frame_000.jpg, frame_001.jpg)
        filename = os.path.join(OUTPUT_DIR, f"frame_{frame_count:03d}.jpg")
        cv2.imwrite(filename, frame)
        
        frame_count += 1
        
        # A tiny sleep to simulate natural frame pacing
        time.sleep(0.03) 

    # 4. Cleanup
    end_time = time.time()
    cap.release()
    
    print("-" * 30)
    print(f"Success! Saved {frame_count} frames in {end_time - start_time:.2f} seconds.")
    print(f"Open the '{OUTPUT_DIR}' folder on your PC to check the images.")

if __name__ == "__main__":
    test_camera_stream()
