# set subjective parameters for UX
PARAMS = {
    'GAIN': 8000,               # higher = slower cursor movement (was 2000)
    'DAMP': 100,               # higher = more stability when holding still (was 60)
    'SENSITIVITY': 15,         # higher = larger "stable" region (was 5)
    'STEPS': 20,               # keep this the same for smooth movement
    'DELAY': 0.0001            # lower = faster response (was 0.0002)
}

# mediapipe landmarks
HAND_LANDMARKS = {
    'MOVE_ID': 9,  # reference point of movement (base of third finger)
    'THUMB_TIP': 4,
    'INDEX_TIP': 8,
    'THUMB_J': 3,  # reference for cliks (joint of thumb)
    'MIDDLE_TIP': 12,
    'RING_TIP': 16,
    'LITTLE_TIP': 20,
    'WRIST': 0,
}

# define virtual frame size in pixels (480x270 is a good tradeoff between resolution and processing speed)
FRAME_SIZE = {'width': 480, 'height': 270}