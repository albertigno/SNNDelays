import cv2
import numpy as np
import random
import time


# Parameters
#symbols = ['@', '#', '&', '∞', '+']  # List of symbols
#symbols = ['@', '#', '&', '+', '%']  # List of symbols
#symbols = ['@', '#', '&', '+', '%']
#symbols = ['I']

#symbols = ['C', 'A', 'P', 'O', 'C', 'A', 'C', 'C', 'I', 'A']

symbols = ['A', 'X', 'O', 'B', 'C']

#symbols = ['X', 'A']

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

# Display loop
cv2.namedWindow("Symbols", cv2.WINDOW_NORMAL)
#cv2.moveWindow("Symbols", 900, 270) # screen 1
cv2.moveWindow("Symbols", 2250, 230) # screen 2
cv2.moveWindow("Symbols", 900, 550) # screen 1
cv2.moveWindow("Symbols", 3150, 700) # screen 3

cv2.resizeWindow("Symbols", 300, 300)

blank_img = np.zeros((300, 300), dtype=np.uint8)

#num_rpts = int(4*frequency) # Number of repetitions for each symbol
num_rpts = 1 # Number of repetitions for each symbol

# count the time
start_time = time.time()

total_symbols = 1e9

# cv2.imshow("Symbols", blank_img)
# # cv2.waitKey(5000)

def record_symbol(symbol):
    """Record the displayed symbol to the log file."""
    with open(log_filename, 'a') as f:
        f.write(symbol)

while total_symbols > 0:
    
    # Randomly shuffle the symbols
    random.shuffle(symbols)

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
            img = np.zeros((300, 300), dtype=np.uint8)

            # Get text size and position (with offset)
            text_size = cv2.getTextSize(symbol, font, font_scale, thickness)[0]
            text_x = (img.shape[1] - text_size[0]) // 2 + offset_x
            text_y = (img.shape[0] + text_size[1]) // 2 + offset_y

            # # Draw rotated text
            # M = cv2.getRotationMatrix2D((text_x, text_y), angle, 1)
            # img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

            cv2.putText(img, symbol, (text_x, text_y), font, font_scale,
                        255, thickness, cv2.LINE_AA)
            cv2.imshow("Symbols", img)

            key = cv2.waitKey(int(duration_per_symbol * 1000))

            cv2.imshow("Symbols", blank_img)

            key = cv2.waitKey(int(blank_duration * 1000))

            if key == 27:  # ESC to quit
                cv2.destroyAllWindows()
                exit()
        
end_time = time.time()
# Calculate total time taken
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")