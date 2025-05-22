import dv_processing as dv
import cv2 as cv
from datetime import timedelta
import numpy as np
from skimage.transform import resize
import torch

from davis_functions import process_image, accumulator
# Open any camera

# from snn_delays.utils.model_loader_refac import ModelLoader
# snn = ModelLoader('ibm_gest_ffw', 'capocaccia_live', 1, 'cpu', live = True)

capture = dv.io.CameraCapture("DAVIS240C_02460013")

# Make sure it supports event stream output, throw an error otherwise
if not capture.isEventStreamAvailable():
    raise RuntimeError("Input camera does not provide an event stream.")

# Initialize preview window
cv.namedWindow("Input", cv.WINDOW_NORMAL)
# Initialize a slicer
slicer = dv.EventStreamSlicer()

all_frames = []

# Declare the callback method for slicer
def slicing_callback(events: dv.EventStore):
    # Pass events into the accumulator and generate a preview frame
    accumulator.accept(events)
    frame = accumulator.generateFrame()

    processed_frame, result = process_image(frame.image, 128)

    result = torch.from_numpy(result).permute(2, 0, 1).unsqueeze(0)

    #all_frames.append(result)
    all_frames.append(processed_frame)

    cv.imshow("Input", processed_frame)

    key = cv.waitKey(1)

    if key & 0xFF == ord('p'):
        while True:
            key = cv.waitKey(0)
            if key == ord('p'):
                break


#time = 600000 # recording time in ms (for some strange reason, we have to set it twice the actual time)

#time = 360000 # 3 minutes
time = 600000 # 5 minutes


time = 100 # 2 seconds

delta_time = 1 # time interval in ms

# Register a callback every 33 milliseconds
slicer.doEveryTimeInterval(timedelta(milliseconds=delta_time), slicing_callback)

# Run the event processing while the camera is connected
#while capture.isRunning():
while capture.isRunning() and time > 0:

    # Receive events
    events = capture.getNextEventBatch()

    # Check if anything was received
    if events is not None:
        time -= delta_time
        print(time)
        # If so, pass the events into the slicer to handle them
        slicer.accept(events)

stacked = np.stack(all_frames)
print(stacked.squeeze().shape)

#np.save('OXS_01.npy', stacked)
#np.save('SOX_01.npy', stacked)
#np.save('randomcaccia.npy', stacked)
np.save('capo_50Hz.npy', stacked)