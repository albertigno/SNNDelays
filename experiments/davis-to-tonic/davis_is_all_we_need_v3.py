import dv_processing as dv
import cv2 as cv
from datetime import timedelta
import numpy as np
from skimage.transform import resize
import torch
import tonic.transforms as transforms
from collections import deque


'''
DAVIS events processed directly by tonic
'''

#from davis_functions import process_image, accumulator

from snn_delays.utils.model_loader_refac import ModelLoader

size = 32  # Size of the input image for the model

slice_time = 5 # Time interval for slicing in milliseconds

snn = ModelLoader('abcxo_f_9783333333333333', 'abcxo_32_24', 1, 'cpu', live = True)

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

letters = ['A', 'B', 'C', 'O', 'X']

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

tonic_dtype = np.dtype([("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)])


sensor_size = (240, 180, 2)
transforms_list = []

# transforms_list.append(
#     transforms.CenterCrop(sensor_size=sensor_size, size=(180, 180)))
# cropped_sensor_size = (180, 180, 2)

transforms_list.append(
    transforms.CenterCrop(sensor_size=sensor_size, size=(128, 128)))
cropped_sensor_size = (128, 128, 2)

target_size = (size, size)
spatial_factor = \
    np.asarray(target_size) / cropped_sensor_size[:-1]

transforms_list.append(
    transforms.Downsample(spatial_factor=spatial_factor[0]))

transforms_list.append(
    transforms.ToFrame(
        sensor_size=(size, size, 2), n_time_bins=1))

to_frame = transforms.Compose(transforms_list)


# create a deque to store the last 10 frames (5 seconds)
saved_frames = deque(maxlen=1000)

# Declare the callback method for slicer
def slicing_callback(events: dv.EventStore):

    # ## for debugging purposes
    # print(f"Received {len(events)} events")
    # print(f"First event: {events.numpy()[0]}")
    # diff = events.numpy()[-1][0] - events.numpy()[0][0]
    # print(f"Time difference: {diff/1000} ms")

    if len(events) > 0:

        events_packets = events.numpy()

        # Extract fields and convert
        x = events_packets['x']
        y = events_packets['y']
        p = events_packets['polarity'].astype(bool)  # Convert to bool
        t = events_packets['timestamp']

        tonic_events = np.empty(len(events_packets), dtype=tonic_dtype)
        tonic_events["x"] = x
        tonic_events["y"] = y
        tonic_events["p"] = p
        tonic_events["t"] = t

        # print(f"Received {len(tonic_events)} tonic events")

        # print(tonic_events)

        frame = to_frame(tonic_events)

        #print(f"Received {len(events)} events")
        #print(f"Frame shape: {frame.shape}")
        # print(f"max: {np.max(frame)}")
        # print(f"min: {np.min(frame)}")

        # type = frame.dtype
        # print(f"Frame type: {type}")
        # print(f"max: {np.max(frame)}")
        # print(f"min: {np.min(frame)}")

        if len(frame.shape)==4:
            frame = (frame>0).astype(np.int16) # Normalize the frame
        else:
            frame = np.zeros((1, 2, size, size)).astype(np.int16)

        saved_frames.append(frame)

        pred = snn.propagate_live(torch.from_numpy(frame))

        #print(pred)

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

        cv.imshow("Input", 255*frame[0, 0, :, :].astype(np.uint8))


# Register a callback every 33 milliseconds
slicer.doEveryTimeInterval(timedelta(milliseconds=slice_time), slicing_callback)

# Run the event processing while the camera is connected
while capture.isRunning():
    # Receive events
    events = capture.getNextEventBatch()

    # Check if anything was received
    if events is not None:
        # print(f"Received {len(events)} events")
        # If so, pass the events into the slicer to handle them
        slicer.accept(events)

        key = cv.waitKey(1)
        if key == ord('q'):
            break

saved_frames = np.array(saved_frames)
print(f"Saved frames shape: {saved_frames.shape}")
np.save('davis_frames_X.npy', saved_frames)