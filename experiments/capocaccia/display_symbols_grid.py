import cv2
import numpy as np
import random

# Step 1: Create a blank canvas and render a character
canvas_size = 200
char = 'A'
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 5
thickness = 10

img = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
text_size = cv2.getTextSize(char, font, font_scale, thickness)[0]
text_x = (canvas_size - text_size[0]) // 2
text_y = (canvas_size + text_size[1]) // 2
cv2.putText(img, char, (text_x, text_y), font, font_scale, 255, thickness, cv2.LINE_AA)

# Step 2: Split into 2x2 grid (4 parts)
h, w = img.shape
half_h, half_w = h // 2, w // 2
quadrants = [
    img[0:half_h, 0:half_w],       # Top-left
    img[0:half_h, half_w:],        # Top-right
    img[half_h:, 0:half_w],        # Bottom-left
    img[half_h:, half_w:]          # Bottom-right
]

# Step 3: Shuffle the order
indices = list(range(4))
random.shuffle(indices)

# Step 4: Create frames showing only one quadrant at a time
frames = []
for idx in indices:
    frame = np.zeros_like(img)
    if idx == 0:
        frame[0:half_h, 0:half_w] = quadrants[0]
    elif idx == 1:
        frame[0:half_h, half_w:] = quadrants[1]
    elif idx == 2:
        frame[half_h:, 0:half_w] = quadrants[2]
    elif idx == 3:
        frame[half_h:, half_w:] = quadrants[3]
    frames.append(frame)

# Step 5: Display or save the sequence
for i, f in enumerate(frames):
    cv2.imshow(f'frame', f)
    cv2.waitKey(500)

cv2.destroyAllWindows()