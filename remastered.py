import mediapipe as mp
import numpy as np
import cv2, time, keyboard, matplotlib, asyncio, threading, sys
matplotlib.use('TkAgg') #backend shit
import multiprocessing as mltp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from bleak import BleakClient, BleakScanner


class Hand_Data:
    def __init__(self):
        self.landmarks = mltp.Array('f', 21 * 3)
        self.landmarks_ready = mltp.Value('b', False)
        self.angles = mltp.Array('i', 16)
        self.angles_ready = mltp.Value('b', False)
        self.palm_normal_vector = mltp.Array('f', 3)
        self.palm_normal_ready = mltp.Value('b', False)

def computer_vision(hand_data):
    front_cam = cv2.VideoCapture(0)
    mp_draw = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    front_hand = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.95, min_tracking_confidence=0.95)
    try:
        while front_cam.isOpened():
            successful, frame = front_cam.read()
            hand_detected = False
            if successful:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                detected_hands = front_hand.process(frame_rgb)
                hand = detected_hands.multi_hand_landmarks[0] if detected_hands.multi_hand_landmarks else None
                if hand:
                    for i, landmark in enumerate(hand.landmark):
                        hand_data.landmarks[i*3:i*3+3] = [landmark.x, landmark.y, landmark.z]
                    hand_detected = True
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                cv2.imshow("Camera", frame)
            hand_data.landmarks_ready.value = hand_detected

            time.sleep(0.001)

            if cv2.waitKey(1) & 0xFF == ord('1'):
                print('Exiting computer Vision')
                break 

    except Exception as e:
        print(f"Exception occured in computer_vision as: {e}")
    
    finally:
        front_cam.release()
        cv2.destroyAllWindows()
        hand_data.landmarks_ready.value = False

# custom object making function for converting hand_data back into landmark.x, 
# landmark.y etc instead of indexing
def get_landmark(hand_data, index):
    start = index * 3
    return type('Landmark', (), {
        'x': hand_data.landmarks[start],
        'y': hand_data.landmarks[start+1],
        'z': hand_data.landmarks[start+2]
    })

def angle_calculations(hand_data):
    finger_dictionary = {
        'Thumb': [4, 3, 2, 1],
        'Index': [8, 7, 6, 5],
        'Middle': [12, 11, 10, 9],
        'Ring': [16, 15, 14, 13],
        'Pinky': [20, 19, 18, 17],
    }

    def calculate_vector(p1, p2):
        return np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

    def get_palm_normal(wrist, index_base, pinky_base):
        wrist_to_index = calculate_vector(wrist, index_base)
        index_to_pinky = calculate_vector(index_base, pinky_base)
        return np.cross(wrist_to_index, index_to_pinky)

    def calculate_angle(v1, v2):
        mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if mag1 == 0 or mag2 == 0:
            return None #check if the return of this is valid to control default angle in lateral and joint
        cosine_angle = np.dot(v1, v2) / (mag1 * mag2)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle

    def calculate_lateral_angle(palm_normal, wrist_to_finger_base, finger_vector):
        lateral_vector = np.cross(palm_normal, wrist_to_finger_base)
        lat_angle = calculate_angle(finger_vector, lateral_vector)
        return lat_angle if lat_angle else None

    def calculate_joint_angle(p1, origin, p3):
        v1, v2 = calculate_vector(origin, p1), calculate_vector(origin, p3)
        angle = calculate_angle(v1, v2)
        return angle if angle else None
    
    def calculate_thumb_forward(angle_b):
        value = abs((2 * angle_b) - 270)
        return max(0, min(90, value))

    try:
        while True:
            angles_ready = False
            if hand_data.landmarks_ready.value:
                landmarks = [get_landmark(hand_data, i) for i in range(21)] #give us all the landmarks
                angles = []

                #initialise palm normal calculation here
                wrist = landmarks[0]
                index_base = landmarks[5]
                pinky_base = landmarks[17]
                palm_normal_vector = get_palm_normal(wrist, index_base, pinky_base)

                hand_data.palm_normal_vector[:] = palm_normal_vector
                hand_data.palm_normal_ready.value = True

                for finger, landmark_list in finger_dictionary.items():
                    current_finger = []
                    for lm in landmark_list:
                        current_finger.append(landmarks[lm])
                    
                    #per finger, calculate joint angles and lateral
                    if all(current_finger):
                        tip, a, b, base = current_finger
                        if finger != 'Thumb': #normal calculation for every other finger
                            angle_a = calculate_joint_angle(tip, a, b) or 180
                            angle_b = calculate_joint_angle(a, b, base) or 0

                            #lateral calculation
                            wrist_to_current_finger_base = calculate_vector(wrist, base)
                            finger_vector = calculate_vector(base, b)
                            lateral = calculate_lateral_angle(palm_normal_vector, 
                                                              wrist_to_current_finger_base,
                                                              finger_vector) or 90
                            
                            angles.extend([int(angle_a), int(angle_b), int(lateral)])

                        else: #thumb 
                            thumb_a = calculate_joint_angle(tip, a, b) or 180
                            thumb_b = calculate_joint_angle(a, b, base) or 0
                            lat = None
                            forward_lat = calculate_thumb_forward(thumb_b)

                            angles.extend([int(thumb_a), int(thumb_b), 90, int(forward_lat)])

                for i, angle in enumerate(angles):
                    hand_data.angles[i] = angle 

                angles_ready = True
                # print(f"angles calculated: {angles}")
            
            if keyboard.is_pressed('2'):
                print("Stopping angle calculation")
                angles_ready = False 
                hand_data.palm_normal_ready.value = False
                break
            
            hand_data.angles_ready.value = angles_ready

            time.sleep(0.01)

        print('Angle calculations stopped')
    
    except Exception as e:
        print(f"An error occured in calculating angles as: {e}")
        hand_data.angles_ready.value = False
        hand_data.palm_normal_ready.value = False


def matplot(hand_data):
    global palm_normal_arrow

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=100, azim=90)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(-0.5, 0.5)

    scatter = ax.scatter([], [], [], c='b', marker='o')
    palm_normal_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='r') 

    finger_dictionary = {
        'Thumb': ([4, 3, 2, 1], 'red'),
        'Index': ([8, 7, 6, 5], 'green'),
        'Middle': ([12, 11, 10, 9], 'blue'),
        'Ring': ([16, 15, 14, 13], 'purple'),
        'Pinky': ([20, 19, 18, 17], 'orange'),
    }

    def update_plot(frame):
        global palm_normal_arrow

        if hand_data.landmarks_ready.value:
            landmarks = [get_landmark(hand_data, i) for i in range(21)]

            x = [lm.x for lm in landmarks]
            y = [lm.y for lm in landmarks]
            z = [lm.z for lm in landmarks]
            scatter._offsets3d = (x, y, z)

            for line in ax.lines:
                line.remove()

            for _, (landmark_list, color) in finger_dictionary.items():
                finger_x = [landmarks[i].x for i in landmark_list]
                finger_y = [landmarks[i].y for i in landmark_list]
                finger_z = [landmarks[i].z for i in landmark_list]
                ax.plot(finger_x, finger_y, finger_z, color=color)
            
            if hand_data.palm_normal_ready.value:
                wrist = landmarks[0]
                palm_normal = np.array(hand_data.palm_normal_vector)
                palm_normal_unit = palm_normal / np.linalg.norm(palm_normal)
                
                palm_normal_arrow.remove()
                palm_normal_arrow = ax.quiver(wrist.x, wrist.y, wrist.z,
                                              palm_normal_unit[0],
                                              palm_normal_unit[1],
                                              palm_normal_unit[2],
                                              color='cyan', arrow_length_ratio=0.1)
            else:
                palm_normal_arrow.remove()
                palm_normal_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='r')

        else:
            scatter._offsets3d = ([], [], [])
            for line in ax.lines:
                line.remove()
            palm_normal_arrow.remove()
            palm_normal_arrow = ax.quiver(0, 0, 0, 0, 0, 0, color='r')

    ani = FuncAnimation(fig, update_plot, interval=10, cache_frame_data=False)
    plt.show()

    while True:
        if keyboard.is_pressed('3'):
            print("Stopping matplot")
            plt.close(fig)
            break

async def ble_session(hand_data, stop_event):
    SERVER_NAME = "ESP-32 S3"
    # MAC_ADDRESS = "68:B6:B3:3E:40:E4"
    # UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
    UART_WRITE_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E" #write to the servers receiving characteristic
    UART_READ_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E" #read from the servers transmission characteristic

    last_data_sent = None 
    local_stop = False

    async def data_is_different(current_data, previous_data):
        if previous_data is None:
            return True
        return any(abs(a - b) >= 2 for a, b in zip(current_data, previous_data))

    print('Scanning for device...')
    device = await BleakScanner.find_device_by_name(SERVER_NAME, timeout=20.0)

    if device is None:
        print("Could not find device")
        return
    
    print(f"{device.name} was found! Attempting to connect...")

    async with BleakClient(device) as client:
        print(f"Connected to {device.name}")

        def notification_handler(sender, data):
            nonlocal local_stop
            if data == b'cunt':
                print(f"{device.name} is shutting down, stopping ble session")
                local_stop = True
        
        await client.start_notify(UART_READ_CHAR_UUID, notification_handler)

        try:
            while not stop_event.is_set() or not local_stop:

                if not client.is_connected:
                    print('Connection lost')
                    local_stop = True
                    stop_event.set()

                if hand_data.angles_ready.value:
                    current_data = list(hand_data.angles)
                    if await data_is_different(current_data, last_data_sent):
                        data_to_send = ','.join(map(str, current_data)).encode()
                        await client.write_gatt_char(UART_WRITE_CHAR_UUID, data_to_send)
                        print(f"sent angles: {current_data}")
                        last_data_sent = current_data 

                await asyncio.sleep(0.01)

                if keyboard.is_pressed('4') or stop_event.is_set() or local_stop:
                    print("Stopping bluetooth session")
                    break

        except Exception as e:
            print(f"An exception occured in ble session as: {e}")

        finally:
            if client.is_connected:
                await client.disconnect()
            print("Disconnected!")
    
    print("BLE session ended")

def run_ble_in_thread(hand_data, stop_event):
    def ble_thread():
        asyncio.set_event_loop(asyncio.new_event_loop())
        asyncio.get_event_loop().run_until_complete(ble_session(hand_data, stop_event))

    ble_thread = threading.Thread(target=ble_thread)
    ble_thread.start()
    return ble_thread

def main():
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    hand_data = Hand_Data()
    ble_stop_event = threading.Event()

    processes = [
        mltp.Process(target=computer_vision, args=(hand_data,)),
        mltp.Process(target=angle_calculations, args=(hand_data,)),
        mltp.Process(target=matplot, args=(hand_data,))
    ]

    for process in processes:
        process.start()

    ble_thread = run_ble_in_thread(hand_data, ble_stop_event)

    try:
        while not keyboard.is_pressed('g'):
            time.sleep(0.01)
    finally:
        ble_stop_event.set()
        for process in processes:
            process.terminate()
            process.join()
        
        ble_thread.join(timeout=5)
        print("All processes and threads have been stopped.")

if __name__ == '__main__':
    main()