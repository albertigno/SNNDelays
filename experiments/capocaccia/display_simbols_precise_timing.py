import cv2
import numpy as np
import random
import time

# Parameters
symbols = ['A']
frequency = 25  # Hz
duration_per_symbol = (1 / frequency)/2  # 20ms for symbol, 20ms for blank
blank_duration = duration_per_symbol 

num_rpts = 8

font_scales = [5, 6, 7, 8]
fonts = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL
]

total_symbols = 50
frame_count = 0

cv2.namedWindow("Symbols", cv2.WINDOW_NORMAL)
cv2.moveWindow("Symbols", 900, 550) # screen 1
cv2.resizeWindow("Symbols", 600, 600)

blank_img = np.zeros((600, 600, 3), dtype=np.uint8)

start_time = time.perf_counter()  # More precise timing

while total_symbols > 0:
    for symbol in symbols:
        rpts = 0
        while rpts < num_rpts:
            total_symbols -= 1
            if total_symbols <= -1:
                break

            rpts += 1
            frame_count += 1
            
            # Measure frame start time
            frame_start = time.perf_counter()
            
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
            img = np.zeros((600, 600, 3), dtype=np.uint8)

            # Get text size and position (with offset)
            text_size = cv2.getTextSize(symbol, font, font_scale, thickness)[0]
            text_x = (img.shape[1] - text_size[0]) // 2 + offset_x
            text_y = (img.shape[0] + text_size[1]) // 2 + offset_y

            # # Draw rotated text
            # M = cv2.getRotationMatrix2D((text_x, text_y), angle, 1)
            # img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

            cv2.putText(img, symbol, (text_x, text_y), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)
            
            # Display symbol
            cv2.imshow("Symbols", img)
            key = cv2.waitKey(1)  # Use minimal wait, we'll handle timing ourselves
            
            # Calculate remaining symbol time
            elapsed = time.perf_counter() - frame_start

            remaining_symbol_time = max(0, duration_per_symbol - elapsed)
            time.sleep(remaining_symbol_time)
            
            # Display blank
            blank_start = time.perf_counter()
            cv2.imshow("Symbols", blank_img)
            key = cv2.waitKey(1)
            
            # Calculate remaining blank time
            elapsed_blank = time.perf_counter() - blank_start
            remaining_blank_time = max(0, blank_duration - elapsed_blank)
            time.sleep(remaining_blank_time)
            
            # [...] (keep your exit condition the same)

end_time = time.perf_counter()

print(duration_per_symbol)
print(elapsed)
print(elapsed_blank)
print(remaining_symbol_time)
print(remaining_blank_time)


total_time = end_time - start_time
actual_frequency = frame_count / total_time
print(f"Total time taken: {total_time:.4f} seconds")
print(f"Actual frequency: {actual_frequency:.2f} Hz")
print(f"Target frequency: {frequency} Hz")
print(f"Deviation: {(actual_frequency - frequency):.2f} Hz ({((actual_frequency - frequency)/frequency*100):.1f}%)")