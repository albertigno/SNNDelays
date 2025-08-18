import cv2
import numpy as np
import random
import time

'''
not showing the whole symbol, but by quadrants 
'''

# Parameters
#symbols = ['@', '#', '&', '∞', '+']  # List of symbols
#symbols = ['@', '#', '&', '+', '%']  # List of symbols
#symbols = ['@', '#', '&', '+', '%']
#symbols = ['I']

#symbols = ['C', 'A', 'P', 'O', 'C', 'A', 'C', 'C', 'I', 'A']

#symbols = ['A', 'X', 'O', 'B', 'C']

symbols = ['A']

log_filename = f"symbol_log.txt"

#symbols = ['P']

frequency = 15  # Hz (symbols will change every 1 second)

duration_per_symbol = (1 / frequency)/2

blank_duration = duration_per_symbol 

font_scales = [5, 6, 7, 8]
fonts = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL
]

size = 300


# Display loop
cv2.namedWindow("Symbols", cv2.WINDOW_NORMAL)
#cv2.moveWindow("Symbols", 900, 270) # screen 1
cv2.moveWindow("Symbols", 2250, 230) # screen 2
cv2.moveWindow("Symbols", 900, 550) # screen 1
cv2.moveWindow("Symbols", 3150, 450) # screen 3

cv2.resizeWindow("Symbols", size, size)

blank_img = np.zeros((size, size), dtype=np.uint8)

#num_rpts = int(4*frequency) # Number of repetitions for each symbol
num_rpts = 1 # Number of repetitions for each symbol

# count the time
start_time = time.time()

total_symbols = 1e6

# cv2.imshow("Symbols", blank_img)
# # cv2.waitKey(5000)

def record_symbol(symbol):
    """Record the displayed symbol to the log file."""
    with open(log_filename, 'a') as f:
        f.write(symbol)

def split_into_quadrants(img):
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

    return frames

while total_symbols > 0:
    
    # Randomly shuffle the symbols
    #random.shuffle(symbols)

    for symbol in symbols:
        
        rpts = 0
            
        while rpts<num_rpts:

            total_symbols -= 1
            if total_symbols <= -1:
                break

            # cv2.waitKey(int(500))

            #record_symbol(symbol)  # Record the symbol to the log file

            rpts += 1
            # pick font randomly
            font = np.random.choice(fonts)
            font_scale = np.random.choice(font_scales)

            # pick thickness randomly
            thickness = random.randint(3, 10)

            # # Random rotation angle (-30 to 30 degrees)
            # angle = random.uniform(45, 46)
            
            # Random position offset (-50 to 50 pixels)
            offset_x = random.randint(-50, 50)
            offset_y = random.randint(-50, 50)

            # for font in fonts:
            img = np.zeros((size, size), dtype=np.uint8)

            # Get text size and position (with offset)
            text_size = cv2.getTextSize(symbol, font, font_scale, thickness)[0]
            text_x = (img.shape[1] - text_size[0]) // 2 + offset_x
            text_y = (img.shape[0] + text_size[1]) // 2 + offset_y

            # # Draw rotated text
            # M = cv2.getRotationMatrix2D((text_x, text_y), angle, 1)
            # img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

            cv2.putText(img, symbol, (text_x, text_y), font, font_scale,
                        255, thickness, cv2.LINE_AA)
            
            frames = split_into_quadrants(img)
            
            for i, f in enumerate(frames):
                cv2.imshow(f'Symbols', f)
                cv2.waitKey(1)            
            cv2.imshow("Symbols", int(0.25*duration_per_symbol * 1000))

            #key = cv2.waitKey(int(duration_per_symbol * 1000))

            cv2.imshow("Symbols", blank_img)

            key = cv2.waitKey(int(blank_duration * 1000))

            if key == 27:  # ESC to quit
                cv2.destroyAllWindows()
                exit()
        
end_time = time.time()
# Calculate total time taken
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")