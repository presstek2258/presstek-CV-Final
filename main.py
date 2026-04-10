import json
import os
import random
import time

import cv2
import requests
from ultralytics import YOLO

YOLO_MODEL = YOLO("./runs/detect/train/weights/best.pt")

# camera stuff
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_X = FRAME_WIDTH / 2
CENTER_Y = FRAME_HEIGHT / 2
MAX_FRAMES_TO_SAVE = 100
NO_OBJECT_COUNTER = 8
W_RANGE = [110, 220]
OUTPUT_DIR = "bbox_camera_frames"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
frame_count = 0

# setup stuff
OBJECTS = ["red cube", "blue cube", "green cube", "container"]
PI_IP = "..."
STREAM_URL = f"http://{PI_IP}:8080/?action=stream"
RPC_URL = f"http://{PI_IP}:9030/"

# car stuff
ROTATION_SPEED = 1  # -2 to 2
MOVE_SPEED = 40  # wheels speed
APPROACH_SPEED = 25  # wheels speed
PIXEL_TO_RADIANS_RATIO = 0.005

# arm stuff
MAX_ERROR = 30
INCREMENTAL_STEP_TIME = 1
DIRECT_STEP_TIME = 0.3
PIXEL_TO_CM_RATIO = 0.03
TIME_TO_RESET_ARM = 0.4  # 0.4 seconds
CLAW_TIME = 0.2  # 0.4 seconds
X_RANGE = [-10.0, 10.0]
Y_RANGE = [12.0, 12.0]
Z_RANGE = [-4.0, 25.0]
X_DEFAULT = 0.0
Y_DEFAULT = 12.0
Z_DEFAULT = 13.0
TILT_PULSE_DEFAULT = 2000
TILT_ID = 3

current_arm_x = X_DEFAULT  # left/right
current_arm_y = Y_DEFAULT  # forwards
current_arm_z = Z_DEFAULT  # height
tilt_pulse = TILT_PULSE_DEFAULT  # starting tilt value


# NOTE: ### COMMUNICATIONS ###


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


# NOTE: ### SERVO ARM FUNCTIONS ###


def camera_sleep(cap, wait_time):
    # same as time.sleep but empties camera buffer
    start_time = time.time()
    while time.time() - start_time < wait_time:
        cap.grab()


def reset_arm(no_wait=False):
    # set initial position
    global current_arm_x, current_arm_y, current_arm_z
    current_arm_x, current_arm_y, current_arm_z = X_DEFAULT, Y_DEFAULT, Z_DEFAULT
    tilt_pulse = TILT_PULSE_DEFAULT
    send_rpc_command("SetServoPosition", [TIME_TO_RESET_ARM, TILT_ID, tilt_pulse])
    send_rpc_command(
        "ArmMoveIk",
        [
            current_arm_x,
            current_arm_y,
            current_arm_z,
            0,
            -90,
            90,
            TIME_TO_RESET_ARM * 1000,
        ],
    )
    if no_wait == False:
        time.sleep(TIME_TO_RESET_ARM)


def move_claw(open=False):
    id = 1
    if open:
        pulse = 2100  # open claw
    else:
        pulse = 1600  # close claw
    send_rpc_command("SetServoPosition", [CLAW_TIME, id, pulse])
    time.sleep(CLAW_TIME + 0.1)


def tilt_look_for_object(cap, object_name):
    global tilt_pulse
    for pulse in range(500, 1300, 100):
        send_rpc_command("SetServoPosition", [CLAW_TIME, TILT_ID, pulse])
        camera_sleep(cap, 0.3)
        bbox = run_yolo_and_save_image_with_bbox(cap, object_name)
        if bbox != None:
            tilt_pulse = pulse
            return bbox
    return None


def run_yolo_and_save_image_with_bbox(cap, object_name):
    global frame_count

    ret, frame = cap.read()
    if not ret:
        return None

    results = YOLO_MODEL(frame, verbose=False, conf=0.3)
    display_frame = results[0].plot()

    # search for boxes
    target_bbox = None
    names = results[0].names
    for box in results[0].boxes:
        class_id = int(box.cls[0].item())
        detected_name = names[class_id]

        if detected_name == object_name:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            target_bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            break

    # save a image with boxes for debugging
    filename = os.path.join(OUTPUT_DIR, f"frame_{frame_count:03d}.jpg")
    print(f"frame_{frame_count:03d}.jpg target bbox: {target_bbox}")
    cv2.imwrite(filename, display_frame)
    frame_count += 1

    return target_bbox


def move_arm_towards_bbox(
    cap,
    object_name,
    bbox=None,
    wait=True,
    target_x=None,
    target_y=None,
    target_z=None,
    incremental_step=False,
    exit_condition1=False,
    exit_condition2=False,
):
    global current_arm_x, current_arm_y, current_arm_z, tilt_pulse

    # take a picture and send back object bbox
    if bbox == None:
        camera_sleep(cap, 0.01)
        bbox = run_yolo_and_save_image_with_bbox(cap, object_name)
        if bbox == None:
            return (False, None)

    if incremental_step:
        step_time = INCREMENTAL_STEP_TIME
    else:
        step_time = DIRECT_STEP_TIME

    x, y, w, h = bbox
    obj_center_x = x + (w / 2)
    obj_center_y = y + (h / 2)

    error_x = obj_center_x - CENTER_X
    error_y = CENTER_Y - obj_center_y  # y axis positive direction is downwards
    print(f"{error_x = }")
    print(f"{error_y = }")

    # cap error to prevent big jumps
    error_x = max(-1 * MAX_ERROR, min(MAX_ERROR, error_x))
    error_y = max(-1 * MAX_ERROR, min(MAX_ERROR, error_y))

    # tilt head to correct for vertical error
    if current_arm_z < 0 or target_z < 0:
        if error_y > 0:
            tilt_pulse = tilt_pulse - 5
        if error_y < 0:
            tilt_pulse = tilt_pulse + 5
        send_rpc_command("SetServoPosition", [step_time, 3, tilt_pulse])

    # map (hor:x, ver:y, depth:z), steps to (hor:x, ver:z, depth:y) steps
    current_arm_x += error_x * PIXEL_TO_CM_RATIO
    current_arm_z += error_y * PIXEL_TO_CM_RATIO

    if target_x != None:
        current_arm_x = target_x
    if target_y != None:
        current_arm_y = target_y
    else:
        current_arm_y = Y_DEFAULT
    if target_z != None:
        current_arm_z = target_z

    # prevents robot from moving outside its safety bounds
    current_arm_x = max(X_RANGE[0], min(X_RANGE[1], current_arm_x))
    current_arm_z = max(Z_RANGE[0], min(Z_RANGE[1], current_arm_z))
    current_arm_y = max(Y_RANGE[0], min(Y_RANGE[1], current_arm_y))

    # to grab object while still
    if (
        exit_condition1
        and abs(error_x) < 25
        and abs(error_y) < 25
        and current_arm_z <= -3
    ):
        print("##### pickup target #####")
        return (True, bbox)

    # to follow the object while moving
    if exit_condition2 and w > 150:
        print("##### arm at target: success #####")
        return (True, bbox)

    print(f"{current_arm_y = }")
    print(f"{current_arm_z = }")

    send_rpc_command(
        "ArmMoveIk",
        [
            current_arm_x,
            current_arm_y,
            current_arm_z,
            0,
            -90,
            90,
            step_time * 1000,
        ],
    )
    if wait:
        time.sleep(step_time + 0.3)
    return (False, bbox)


# NOTE: ### MECHANUM ###


def mechanum_360_obj_detection(cap, object_name):
    # spins the car 360 to find the object_name
    # stops spinning when the object is in frame
    if object_name == "container":
        rotation_speed = ROTATION_SPEED
    else:
        rotation_speed = -1 * ROTATION_SPEED

    while True:
        bbox = tilt_look_for_object(cap, object_name)
        if bbox != None:
            return bbox

        # approach target
        send_rpc_command("SetChassisVelocity", [0, 0, rotation_speed])
        time.sleep(0.10)
        send_rpc_command("SetChassisVelocity", [0, 0, 0])
        camera_sleep(cap, 1)  # special sleep to flush mjpg buffer


def mechanum_center_car_on_target(cap, object_name, first_bbox=None):
    no_object_counter = 0
    while True:
        print(f"Centering:{no_object_counter + 1}...")
        if no_object_counter == NO_OBJECT_COUNTER / 2:
            print("mechanum centered on target: failed")
            return None

        if first_bbox == None:
            camera_sleep(cap, 0.5)
            bbox = run_yolo_and_save_image_with_bbox(cap, object_name)
            if bbox == None:
                no_object_counter += 1
                continue
        else:
            bbox = first_bbox
            first_bbox = None

        # center the car on the bbox
        x, y, w, h = bbox
        obj_center_x = x + (w / 2)
        error_x = obj_center_x - CENTER_X

        if abs(error_x) < 35:
            print("##### mechanum centered on target: success #####")
            return bbox

        # map (hor:x, ver:y, depth:z), steps to (hor:x, ver:z, depth:y) steps
        radians = error_x * PIXEL_TO_RADIANS_RATIO
        radians = max(-1.5, min(1.5, radians))

        send_rpc_command("SetChassisVelocity", [0, 0, radians])
        time.sleep(0.05)
        send_rpc_command("SetChassisVelocity", [0, 0, 0])


# NOTE: ### RUNNER FUNCTIONS ###


def main():
    try:
        cap = cv2.VideoCapture(STREAM_URL)
        if not cap.isOpened():
            return
        object_names = ["red cube", "blue cube", "green cube"]
        random.shuffle(object_names)
        object_name_index = 0
        object_name = object_names[object_name_index]
        print(f"object_name = {object_names[object_name_index]}")

        # object detect and sort loop
        while True:
            # NOTE: find and center car on object
            reset_arm()
            move_claw(open=False)
            while True:
                bbox = mechanum_360_obj_detection(cap, object_name)
                bbox = mechanum_center_car_on_target(cap, object_name, first_bbox=bbox)
                if bbox != None:
                    break

            # NOTE: approach until adjacent to object
            while True:
                if bbox == None:
                    break
                x, y, w, h = bbox

                # check if centering is required
                obj_center_x = x + (w / 2)
                error_x = obj_center_x - CENTER_X

                if abs(error_x) > 100:
                    bbox = None  # this will cause continue to trigger after break
                    break

                print(f"{w = }")
                print(f"{y = }")
                if object_name == "container":
                    w_range = [300, 800]
                else:
                    w_range = W_RANGE

                if w >= w_range[0] and w <= w_range[1] and y > 130:
                    break
                if w > w_range[1] or y > FRAME_HEIGHT - 100:
                    send_rpc_command("SetChassisVelocity", [-1 * MOVE_SPEED, 90, 0])
                    time.sleep(0.3)
                else:
                    if w > 80:
                        send_rpc_command("SetChassisVelocity", [APPROACH_SPEED, 90, 0])
                        time.sleep(0.2)
                    else:
                        send_rpc_command("SetChassisVelocity", [MOVE_SPEED, 90, 0])
                        time.sleep(0.4)

                # inch car forward and look for object
                send_rpc_command("SetChassisVelocity", [0, 0, 0])
                camera_sleep(cap, 0.1)
                bbox = tilt_look_for_object(cap, object_name)
            if bbox == None:
                continue
            print(f"##### beside {object_name}#####")

            # NOTE: if holding an object place it in the container
            if object_name == "container":
                print("##### direct move to place object #####")
                centered, bbox = move_arm_towards_bbox(
                    cap,
                    object_name,
                    bbox=bbox,
                    wait=False,
                    target_z=-1,
                    incremental_step=False,
                )
                move_claw(open=True)
                time.sleep(0.3)
                reset_arm()
                move_claw(open=False)
                send_rpc_command("SetChassisVelocity", [-1 * MOVE_SPEED, 90, 0])
                time.sleep(0.3)
                send_rpc_command("SetChassisVelocity", [0, 0, 0])
                object_name_index += 1
                if len(object_names) <= object_name_index:
                    return None
                    print("##### all objects in container #####")
                object_name = object_names[object_name_index]
                continue

            # NOTE: move claw/arm to object
            print("##### stepping claw/arm to object #####")
            move_claw(open=True)
            no_object_counter = 0
            object_detected_counter = 0
            y = Y_DEFAULT
            z = Z_DEFAULT
            while True:
                if no_object_counter >= NO_OBJECT_COUNTER * 3:
                    if z >= 0:
                        reset_arm()
                        no_object_counter = 0
                        object_detected_counter = 0
                        bbox = tilt_look_for_object(cap, object_name)
                    else:
                        break

                if object_detected_counter >= 6:
                    break

                centered, bbox = move_arm_towards_bbox(
                    cap,
                    object_name,
                    bbox=None,
                    wait=False,
                    target_z=z,
                    incremental_step=True,
                    exit_condition1=True,
                )
                z -= 3
                if centered:
                    break
                if bbox == None:
                    no_object_counter += 1
                    print(f"Lost target... (object_counter = {no_object_counter})")
                if bbox != None and z < -1:
                    object_detected_counter += 1
                    print(
                        f"Found target... (object_detected_counter = {object_detected_counter})"
                    )

            # NOTE: pickup object and move backwards
            move_claw(open=False)
            reset_arm()
            send_rpc_command("SetChassisVelocity", [-1 * MOVE_SPEED, 90, 0])
            time.sleep(1.5)
            send_rpc_command("SetChassisVelocity", [0, 0, 0])

            # NOTE: go drop the object in the container
            print("##### changing object name #####")
            object_name = "container"
        # LOOP END

    except KeyboardInterrupt:
        cap.release()
        send_rpc_command("SetChassisVelocity", [0, 0, 0])
        reset_arm()
        move_claw(open=False)
        print("interrupted")
    finally:
        cap.release()
        send_rpc_command("SetChassisVelocity", [0, 0, 0])
        reset_arm()
        move_claw(open=False)
        print("done")


if __name__ == "__main__":
    main()
