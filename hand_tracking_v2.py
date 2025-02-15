'''
Hand tracking script acting with full keyboard + mouse controls
Call script directly only when processing camera feed on a separate machine (e.g. Raspberry Pi)
'''

import cv2
import mediapipe as mp
import math
import asyncio
from concurrent.futures import ThreadPoolExecutor
import struct
import time
from config import HAND_LANDMARKS, FRAME_SIZE

# define RUN_MODE
RUN_MODE = "serial" if __name__ == "__main__" else "async"
print(f"Running in {RUN_MODE} mode")

# initialise mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# initialize camera
cap = cv2.VideoCapture(0)  # Remove V4L2 backend specification
# optimise camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE['width'])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE['height'])
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # low latency

### optional: open camera in fullscreen
# window_name = "Hand Tracking"
# cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

executor = ThreadPoolExecutor()


def dist(lm1, lm2, w, h):
    """Calculate Euclidian distance between 2 landmarks"""

    dx = (lm1.x - lm2.x) * w
    dy = (lm1.y - lm2.y) * h
    return math.sqrt(dx ** 2 + dy ** 2)


async def process_frame(frame_queue, landmark_queue):
    """Process each camera frame to track hand movements"""

    # calculate real FPS
    frame_count = 0
    start_time = time.time()

    while True:
        frame = await frame_queue.get()
        if frame is None:
            break

        # Increment frame count
        frame_count += 1

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Calculate FPS every second
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            #print(f"FPS: {fps:.2f}")
            frame_count = 0  # Reset frame count
            start_time = time.time()  # Reset start time

        # Convert frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get landmarks via mediapipe and append to Asyncio queue
        results = hands.process(rgb_frame)
        
        #print("Hand detected:", bool(results.multi_hand_landmarks))  # Debug print
        
        # Put both frame and results in the landmark queue
        await landmark_queue.put((frame, results))


async def send_data(landmark_queue, data_queue, serial_port):
    """
    RUN_MODE = serial: sends data packets over serial to be read by control_machine.py
    RUN_MODE = async: appends data packets to queues to be read by control_machine.py
    """

    while True:
        try:
            frame, results = await landmark_queue.get()
            
            if results.multi_hand_landmarks:
                # print("Processing hand landmarks")  # Debug print
                for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = 'R' if hand_info.classification[0].label == "Left" else 'L'
                    # print(f"Hand detected: {hand_label}")  # Debug print

                    # calculate hand size
                    HAND_SIZE = dist(
                        hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                        hand_landmarks.landmark[HAND_LANDMARKS['MOVE_ID']],
                        FRAME_SIZE['width'], FRAME_SIZE['height'])
                    
                    # CASE 1: Check if in scrolling mode
                    if (
                            HAND_SIZE/2 <
                            dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                                 hand_landmarks.landmark[HAND_LANDMARKS['INDEX_TIP']],
                                 FRAME_SIZE['width'], FRAME_SIZE['height']) and
                            HAND_SIZE/2 <
                            dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                                 hand_landmarks.landmark[HAND_LANDMARKS['MIDDLE_TIP']],
                                 FRAME_SIZE['width'], FRAME_SIZE['height']) and
                            HAND_SIZE >
                            dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                                 hand_landmarks.landmark[HAND_LANDMARKS['RING_TIP']],
                                 FRAME_SIZE['width'], FRAME_SIZE['height']) and
                            HAND_SIZE >
                            dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                                 hand_landmarks.landmark[HAND_LANDMARKS['LITTLE_TIP']],
                                 FRAME_SIZE['width'], FRAME_SIZE['height'])
                    ):
                        print("Scroll mode detected")  # Debug print
                        # reference for scroll movement = tip of index finger
                        scroll_loc = hand_landmarks.landmark[HAND_LANDMARKS['INDEX_TIP']]
                        # reference for scroll anchor = MOVE_ID (base of middle finger)
                        anchor_loc = hand_landmarks.landmark[HAND_LANDMARKS['MOVE_ID']]

                        # normalise coord and flip axis
                        scroll_loc = 1.0 - scroll_loc.y
                        anchor_loc = 1.0 - anchor_loc.y

                        # Clamp values between 0 and 1000
                        scroll_loc = max(0, min(1.0, scroll_loc))
                        anchor_loc = max(0, min(1.0, anchor_loc))

                        # scale float to integer for efficient sending over serial
                        scroll_loc = int(scroll_loc * 1000)
                        anchor_loc = int(anchor_loc * 1000)

                        # binary encode the data for sending over serial with no padding
                        # 6 bytes = 1 char (S for scrolling) + 2 int (scroll and move y-locations) + newline
                        data = struct.pack('=c2H', b'S', scroll_loc, anchor_loc) + b'\n'

                        # transmit data depending on mode
                        if RUN_MODE == "serial":
                            serial_port.write(data)
                        else:
                            print(f"Sending scroll data to queue: {data}")  # Debug print
                            await data_queue.put(data)

                    # CASE 2: cursor mode
                    else:
                        # Get cursor position
                        loc = hand_landmarks.landmark[HAND_LANDMARKS['MOVE_ID']]
                        x_loc, y_loc = 1.0 - loc.x, 1.0 - loc.y

                        # Clamp values between 0 and 1000 before converting to integers
                        x_loc = max(0, min(1.0, x_loc))
                        y_loc = max(0, min(1.0, y_loc))

                        # Convert to integers (0-1000 range)
                        x_loc = int(x_loc * 1000)
                        y_loc = int(y_loc * 1000)

                        data = struct.pack('=c2H', hand_label.encode(), x_loc, y_loc) + b'\n'
                        if RUN_MODE == "serial":
                            serial_port.write(data)
                        else:
                            # print(f"Sending cursor data to queue: {data}")  # Debug print
                            await data_queue.put(data)

                        # Check for click
                        THRESH = dist(
                            hand_landmarks.landmark[HAND_LANDMARKS['THUMB_TIP']],
                            hand_landmarks.landmark[HAND_LANDMARKS['THUMB_J']],
                            FRAME_SIZE['width'], FRAME_SIZE['height'])
                        click = dist(
                            hand_landmarks.landmark[HAND_LANDMARKS['THUMB_TIP']],
                            hand_landmarks.landmark[HAND_LANDMARKS['INDEX_TIP']],
                            FRAME_SIZE['width'], FRAME_SIZE['height'])

                        if THRESH > click:
                            # print("Click detected!")  # Debug print
                            if RUN_MODE == "serial":
                                serial_port.write(b'C\n')
                            else:
                                await data_queue.put(b'C\n')

                        ## CASE 2.2 -> exit (= close fist)
                        if (
                                HAND_SIZE >
                                dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                                     hand_landmarks.landmark[HAND_LANDMARKS['INDEX_TIP']],
                                     FRAME_SIZE['width'], FRAME_SIZE['height']) and
                                HAND_SIZE >
                                dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                                     hand_landmarks.landmark[HAND_LANDMARKS['MIDDLE_TIP']],
                                     FRAME_SIZE['width'], FRAME_SIZE['height']) and
                                HAND_SIZE/2 >
                                dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                                     hand_landmarks.landmark[HAND_LANDMARKS['RING_TIP']],
                                     FRAME_SIZE['width'], FRAME_SIZE['height']) and
                                HAND_SIZE/2 >
                                dist(hand_landmarks.landmark[HAND_LANDMARKS['WRIST']],
                                     hand_landmarks.landmark[HAND_LANDMARKS['LITTLE_TIP']],
                                     FRAME_SIZE['width'], FRAME_SIZE['height'])
                        ):
                            # send 1 byte
                            if RUN_MODE == "serial":
                                serial_port.write(b'E\n')
                            else:
                                await data_queue.put(b'E\n')

                        ## CASE 2.3 -> change tab forward
                        tabf = dist(
                            hand_landmarks.landmark[HAND_LANDMARKS['THUMB_TIP']],
                            hand_landmarks.landmark[HAND_LANDMARKS['RING_TIP']],
                            FRAME_SIZE['width'], FRAME_SIZE['height'])

                        if THRESH > tabf:
                            # send 1 byte
                            if RUN_MODE == "serial":
                                serial_port.write(b'F\n')
                            else:
                                await data_queue.put(b'F\n')

                        ## CASE 2.4 -> change tab backward
                        tabb = dist(
                            hand_landmarks.landmark[HAND_LANDMARKS['THUMB_TIP']],
                            hand_landmarks.landmark[HAND_LANDMARKS['MIDDLE_TIP']],
                            FRAME_SIZE['width'], FRAME_SIZE['height'])

                        if THRESH > tabb:
                            # send 1 byte
                            if RUN_MODE == "serial":
                                serial_port.write(b'B\n')
                            else:
                                await data_queue.put(b'B\n')

                        ## CASE 2.5 -> mission control
                        tabb = dist(
                            hand_landmarks.landmark[HAND_LANDMARKS['THUMB_TIP']],
                            hand_landmarks.landmark[HAND_LANDMARKS['LITTLE_TIP']],
                            FRAME_SIZE['width'], FRAME_SIZE['height'])

                        if THRESH > tabb:
                            # send 1 byte
                            if RUN_MODE == "serial":
                                serial_port.write(b'M\n')
                            else:
                                await data_queue.put(b'M\n')

        except Exception as e:
            print(f"Error in send_data: {e}")
            print(f"Error details:", str(e.__class__), str(e))  # More detailed error info


async def main(data_queue=None):
    """Main event loop"""

    # initialize serial communication conditionally
    if RUN_MODE == "serial":
        import serial
        ## EDIT PORT based to your camera input
        serial_port = serial.Serial('/dev/ttyGS0', 115200, timeout=1)
    else:
        serial_port = None

    frame_queue = asyncio.Queue()               # stores camera frames
    landmark_queue = asyncio.Queue()            # stores landmarks within the frames

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    # Create window for display
    cv2.namedWindow("Hand Tracking")

    # create and immediately run tasks
    async with asyncio.TaskGroup() as tg:
        tg.create_task(process_frame(frame_queue, landmark_queue))
        tg.create_task(send_data(landmark_queue, data_queue, serial_port))

        while cap.isOpened():
            # send to run on separate thread (reading frames is blocking process)
            ret, frame = await asyncio.get_event_loop().run_in_executor(executor, cap.read)
            if not ret:
                print("Failed to grab frame")
                break

            # attach frame to queue for processing
            await frame_queue.put(frame)

            try:
                # Display the frame with hand landmarks
                mirror = cv2.flip(frame, 1)  # Mirror the frame horizontally
                cv2.imshow("Hand Tracking", mirror)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting...")
                    break
                    
            except Exception as e:
                print(f"Display error: {e}")

    print("Cleaning up...")
    # stop processes
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())