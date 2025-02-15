'''
Translates data from serial into mouse and keyboard actions
3 categories: cursor movement, scroll, commands
Call script directly only when processing camera feed on a separate machine (e.g. Raspberry Pi)
'''

from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key
import keyboard
import pyautogui
from screeninfo import get_monitors
from collections import deque
import asyncio
import time
import sys
import struct
from config import PARAMS

# define RUN_MODE
RUN_MODE = "serial" if __name__ == "__main__" else "async"

# Initialize
mouse = MouseController()
pykeyboard = KeyboardController()
monitors = get_monitors()
primary_monitor = monitors[0]
SCREEN_WIDTH = primary_monitor.width
SCREEN_HEIGHT = primary_monitor.height

# Smoothing buffers for moving average
buffer_size = 3
x_buffer = deque(maxlen=buffer_size)
y_buffer = deque(maxlen=buffer_size)

# initialise mouse clicks / position
last_click = 0
cooldown = 0.5          # seconds
scroll_anchor = None

# At the top of the file, after imports
print("Loaded PARAMS:", PARAMS)  # Debug print

########
class StopException(Exception):
    """Custom exception to signal a graceful shutdown."""
    pass

def map_to_screen(loc):
    """Map integer coordinates received over serial (in range 0->1000) to screen coordinates"""
    def zoom(value, screen_size):
        if value < 200:
            return 0
        elif value > 800:
            return screen_size
        else:
            return int(((value - 200) / 600) * screen_size)

    screen_x = int(zoom(loc[0], SCREEN_WIDTH))
    screen_y = int(zoom(loc[1], SCREEN_HEIGHT))
    return [screen_x, screen_y]

def velocity_scale(cur, tar, GAIN=PARAMS['GAIN'], DAMP=PARAMS['DAMP'], SENSITIVITY=PARAMS['SENSITIVITY'], MIN_STEP=1):
    """Adjust speed of cursor based on distance"""
    # calculate Euclidian distance
    distance = ((tar[0] - cur[0]) ** 2 + (tar[1] - cur[1]) ** 2) ** 0.5

    # apply damping
    damping = max(1, SENSITIVITY / max(distance, 1e-6))
    damping *= DAMP if damping > 1 else 1

    # calculate scaling factor
    if distance < SENSITIVITY:
        scaling_factor = MIN_STEP + (distance * damping)
    else:
        scaling_factor = MIN_STEP + (distance / GAIN) * damping

    # calculate steps
    dx = (tar[0] - cur[0]) / scaling_factor
    dy = (tar[1] - cur[1]) / scaling_factor

    # calculate new positions
    new = [cur[0] + dx, cur[1] + dy]

    # Move cursor
    if distance > SENSITIVITY:
        interpolate(cur, new)
    else:
        mouse.position = (int(new[0]), int(new[1]))

    return new

def lerp(start, end, factor):
    """Linear interpolation between start (current) and end (target) points"""
    return start + (end - start) * factor

def interpolate(start, end, steps=PARAMS['STEPS'], delay=PARAMS['DELAY']):
    """
    Interpolates between current and target positions to fill the visual gaps of the cursor
    More steps = smoother mouse cursor but more perceived lag
    """
    for i in range(1, steps + 1):
        # Interpolate between start and end positions
        interp_x = lerp(start[0], end[0], i / steps)
        interp_y = lerp(start[1], end[1], i / steps)

        # Move the mouse to the interpolated position
        mouse.position = (int(interp_x), int(interp_y))

        # Small delay to ensure smooth visual movement
        time.sleep(delay)

async def read_serial(serial_reader, data_queue):
    """
    Read data asynchronously from the serial port
    Protocol =
    2 bytes for command (1 char + newline)
    6 bytes for scroll (1 char + 2 int + newline)
    6 bytes for cursor movement (1 char + 2 int + newline)
    """
    buffer = b''  # store packets as they come in

    while True:
        start_read = time.time()

        try:
            # read 1 byte of data and immediately append to buffer
            chunk = await serial_reader.read(1)
            if not chunk:
                continue  # Skip if no data is received

            buffer += chunk

            # Process complete packets (ending with newline)
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)  # Split at the first newline
                if len(line) > 0:
                    await data_queue.put(line)  # Queue the complete packet

                    end_read = time.time()
                    # print(f"Time to process 1 packet: {end_read - start_read:.6f} seconds")

        except Exception as e:
            print(f"Error reading serial data: {e}")
            break


async def process_data(data_queue, cur):
    """Process serial data and perform cursor actions"""
    global last_click, scroll_anchor

    while True:
        try:
            # Get the next packet from the queue
            data = await data_queue.get()

            # Handle command packets (length 2 - includes newline character)
            if len(data) == 2:
                command = data[0:1]  # Get just the command byte
                if command == b'C':
                    current_time = time.time()
                    if current_time - last_click > cooldown:
                        mouse.click(Button.left)
                        last_click = current_time
                elif command == b'E':
                    raise StopException()
                elif command == b'F':
                    with pykeyboard.pressed(Key.ctrl):
                        pykeyboard.press(Key.tab)
                        pykeyboard.release(Key.tab)
                elif command == b'B':
                    with pykeyboard.pressed(Key.ctrl):
                        with pykeyboard.pressed(Key.shift):
                            pykeyboard.press(Key.tab)
                            pykeyboard.release(Key.tab)
                elif command == b'M':
                    pyautogui.keyDown("ctrl")
                    pyautogui.press("up")
                    pyautogui.keyUp("ctrl")
                continue

            # Handle movement packets (length 6 - includes newline character)
            if len(data) == 6:
                try:
                    command = data[0:1]

                    if command == b'S':
                        # Unpack binary data (ignore newline)
                        _, scroll_loc, anchor_loc = struct.unpack('=c2H', data[:-1])
                        
                        # Flip y-axis
                        scroll_loc = 1000 - int(scroll_loc)
                        anchor_loc = 1000 - int(anchor_loc)

                        # set scroll anchor (relative to hand position)
                        if scroll_anchor is None:
                            scroll_anchor = anchor_loc

                        scroll_y = int((scroll_anchor - scroll_loc) / 10)
                        mouse.scroll(dx=0, dy=scroll_y)

                    elif command in [b'R', b'L']:
                        scroll_anchor = None

                        # Unpack binary data (ignore newline)
                        hand_label, x_loc, y_loc = struct.unpack('=c2H', data[:-1])

                        # Flip y-axis
                        loc = [int(x_loc), 1000 - int(y_loc)]

                        # Convert to screen coordinates
                        tar = map_to_screen(loc)

                        # Get current mouse position
                        cur_pos = list(mouse.position)

                        # Move cursor with velocity scaling
                        new_pos = velocity_scale(cur_pos, tar)
                        
                        # Update current position
                        cur = new_pos

                except Exception as e:
                    print(f"Error processing movement data: {e}")

        except StopException:
            break
        except Exception as e:
            print(f"Error in process_data main loop: {e}")


async def main(data_queue=None):
    """Main event loop"""

    print("Listening for data from Hand Tracking script...")

    # set initial cur_x, cur_y
    cur = [0,0]

    # get data_queue from hand_tracking script if in async mode
    if RUN_MODE == "async" and data_queue is not None:
        try:
            await process_data(data_queue, cur)

        except StopException:
            # if "stop" received, shut down program gracefully
            print("PROGRAM ENDED")

    # initialize data_queue if in serial mode
    elif RUN_MODE == "serial":
        import serial_asyncio
        serial_port = await serial_asyncio.open_serial_connection(url='/dev/tty.usbmodem14101', baudrate=115200)
        reader, writer = serial_port

        data_queue = asyncio.Queue()

        try:
            # create and run tasks for reading and processing data
            async with asyncio.TaskGroup() as tg:
                tg.create_task(read_serial(reader, data_queue))
                tg.create_task(process_data(data_queue, cur))

        except StopException:
            # if "stop" received, shut down program gracefully
            print("PROGRAM ENDED")
    else:
        print("Invalid RUN_MODE or missing data_queue")


if __name__ == "__main__":
    # After initializing mouse controller
    print("Testing mouse controller...")
    try:
        original_pos = mouse.position
        print(f"Original position: {original_pos}")
        test_pos = (100, 100)
        mouse.position = test_pos
        print(f"Moved to test position: {test_pos}")
        time.sleep(1)
        mouse.position = original_pos
        print("Mouse controller test successful")
    except Exception as e:
        print(f"Mouse controller test failed: {e}")

    asyncio.run(main())