# SCRIPT WRITTEN BY GEMINI 3 PRO
import os
import sys
import time

import cv2
import requests

# --- Configuration ---
PI_IP = "..."
STREAM_URL = f"http://{PI_IP}:8080/?action=stream"
RPC_URL = f"http://{PI_IP}:9030/"
IMAGES_TO_COLLECT = 10  # Adjust this to your liking
TIME_BETWEEN_SHOTS = 5


def send_rpc_command(method, params, max_retries=10):
    # sends a movement command to the MasterPi
    payload = {"method": method, "params": params, "jsonrpc": "2.0", "id": 0}
    for attempt in range(max_retries):
        try:
            response = requests.post(RPC_URL, json=payload, timeout=2.0)
            if response.status_code == 200:
                return response.json()
            else:
                print(
                    f"HTTP server error {response.status_code}: {response.text[:200]}"
                )

        except requests.exceptions.RequestException as e:
            print(f"network error: {e}. retrying... {attempt+1}/{max_retries}")
        time.sleep(0.05)

    print(f"failed to execute rpc after {max_retries} attempts")
    return None


def camera_sleep(cap, wait_time):
    """Actively flushes the MJPEG buffer so the next read is completely live."""
    start_time = time.time()
    while time.time() - start_time < wait_time:
        cap.grab()


def main():
    send_rpc_command("ArmMoveIk", [0, 12.0, 13.0, 0, -90, 90, 0.4 * 1000])

    print("=== YOLO Data Collector ===")
    print("Available objects: red_cube, blue_cube, green_cube, container, combo, pair")
    object_name = (
        input("Type the name of the object you are photographing: ")
        .strip()
        .replace(" ", "_")
    )

    # Create a dedicated directory for this object
    output_dir = os.path.join("dataset_raw", object_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("Failed to connect to the MasterPi camera stream.")
        return

    print(f"\nStarting collection for '{object_name}' in 3 seconds...")
    time.sleep(5)

    for i in range(IMAGES_TO_COLLECT):
        # 1. Wait 2 seconds while flushing old frames
        camera_sleep(cap, TIME_BETWEEN_SHOTS)

        # buzz
        send_rpc_command("UseBuzzer", [])

        # 2. Grab the live, crisp frame
        ret, frame = cap.read()
        if not ret:
            print("Dropped a frame, skipping...")
            continue

        # 3. Save the image
        unique_id = int(time.time() * 1000)
        filename = os.path.join(output_dir, f"{object_name}_{unique_id}.jpg")
        cv2.imwrite(filename, frame)

        # 4. Trigger PC Beep and console output
        sys.stdout.write("\a")
        sys.stdout.flush()
        print(f"[{i+1}/{IMAGES_TO_COLLECT}] SNAP! Saved {filename}")

    cap.release()
    print(f"\nCollection complete! Images saved to {output_dir}")


if __name__ == "__main__":
    main()
