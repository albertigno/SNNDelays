import dv_processing as dv
import cv2 as cv
from datetime import timedelta
import numpy as np
from skimage.transform import resize
import torch

from davis_functions import process_image, accumulator
#from cool_vizualizer_tools import draw_mems, draw_spks
# Open any camera

from snn_delays.utils.model_loader_refac import ModelLoader
#snn = ModelLoader('ibm_gest_ffw', 'capocaccia_live', 1, 'cpu', live = True)

snn = ModelLoader('capo1_f_7329608938547486', 'capocaccia', 1, 'cpu', live = True)

capture = dv.io.CameraCapture("DAVIS240C_02460013")

# Make sure it supports event stream output, throw an error otherwise
if not capture.isEventStreamAvailable():
    raise RuntimeError("Input camera does not provide an event stream.")


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
          (255, 0, 255)]  # Unique colors

# Initialize preview window
# cv.namedWindow("Positive", cv.WINDOW_NORMAL)
# cv.namedWindow("Negative", cv.WINDOW_NORMAL)
cv.namedWindow("Input", cv.WINDOW_NORMAL)
# Initialize a slicer
slicer = dv.EventStreamSlicer()

letters = ['C', 'A', 'P', 'O', 'I']

# Settings (adjust as needed)
WINDOW_NAME = "Live Prediction"
WIDTH, HEIGHT = 400, 200  # Small window = faster rendering
FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 5
FONT_THICKNESS = 5
BG_COLOR = (0, 0, 0)  # Black background
TEXT_COLOR = (255, 255, 255)  # White text

# Initialize window (fullscreen optional for visibility)
cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)  
cv.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

# Blank canvas (reused for efficiency)
canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

# Declare the callback method for slicer
def slicing_callback(events: dv.EventStore):
    # Pass events into the accumulator and generate a preview frame
    accumulator.accept(events)
    frame = accumulator.generateFrame()

    #processed_frame, result = process_image(frame.image, 32) # gestures32
    processed_frame, result = process_image(frame.image, 128)

    result = torch.from_numpy(result).permute(2, 0, 1).unsqueeze(0)

    print(torch.min(result))
    print(torch.max(result))

    pred = snn.propagate_live(result) 

    print(pred)

    prediction = letters[pred]

    #draw_mems(snn.mems_fifo['output'], colors)
    #draw_spks(snn.spikes_fifo['l2'], 64)

    canvas[:] = BG_COLOR

    # Calculate text position (centered)
    (text_width, text_height), _ = cv.getTextSize(
        prediction, FONT, FONT_SCALE, FONT_THICKNESS
    )
    x = (WIDTH - text_width) // 2
    y = (HEIGHT + text_height) // 2

    # Draw the prediction
    cv.putText(
        canvas, prediction, (x, y),
        FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS
    )

    # Display (no delay for max speed, or use `cv2.waitKey(1)` for ~1000FPS)
    cv.imshow(WINDOW_NAME, canvas)

    # # Exit on 'q' key press
    # if cv.waitKey(1) & 0xFF == ord('q'):
    #     break


    # Show the accumulated image
    #cv.imshow("Positive", result[0, 0, :, :].cpu().numpy())
    #cv.imshow("Negative", result[0, 1, :, :].cpu().numpy())
    cv.imshow("Input", processed_frame)

    key = cv.waitKey(1)

    if key & 0xFF == ord('p'):
        while True:
            key = cv.waitKey(0)
            if key == ord('p'):
                break

# Register a callback every 33 milliseconds
slicer.doEveryTimeInterval(timedelta(milliseconds=20), slicing_callback)

# Run the event processing while the camera is connected
while capture.isRunning():
    # Receive events
    events = capture.getNextEventBatch()

    # Check if anything was received
    if events is not None:
        # If so, pass the events into the slicer to handle them
        slicer.accept(events)



label = 0 # circle
# label = 1 # cross
# label = 2 # wave
# label = 3 # square
# label = 4 # triangle
