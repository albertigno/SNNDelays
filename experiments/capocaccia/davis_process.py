import dv_processing as dv
import cv2 as cv
from datetime import timedelta
import numpy as np
from skimage.transform import resize
# Open any camera

def process_image(img, final_size=32):
    # 1. Central crop to (180, 180)
    h, w = img.shape
    crop_size = 180
    start_x = (w - crop_size) // 2
    cropped = img[:, start_x:start_x + crop_size]

    # 2. Downsample to (final_size, final_size)
    downsampled = resize(cropped, (final_size, final_size), mode='reflect', anti_aliasing=True)

    # # 3. Split into positive and negative channels
    pos_channel = np.clip(downsampled, 0.5, None)  # keep only positives
    neg_channel = np.clip(downsampled , None, 0.5)  # keep only negatives (in positive scale)

    # # 4. Stack channels -> shape: (final_size, final_size, 2)
    result = np.stack((pos_channel, neg_channel), axis=-1)

    return downsampled, result

capture = dv.io.CameraCapture("DAVIS240C_02460013")

# Make sure it supports event stream output, throw an error otherwise
if not capture.isEventStreamAvailable():
    raise RuntimeError("Input camera does not provide an event stream.")

# Initialize an accumulator with some resolution
accumulator = dv.Accumulator(capture.getEventResolution())

# Apply configuration, these values can be modified to taste
accumulator.setMinPotential(0.0)
accumulator.setMaxPotential(1.0)
accumulator.setNeutralPotential(0.5)
accumulator.setEventContribution(0.15)
accumulator.setDecayFunction(dv.Accumulator.Decay.STEP)
accumulator.setDecayParam(1e3)
accumulator.setIgnorePolarity(False)
accumulator.setSynchronousDecay(False)

# Initialize preview window
cv.namedWindow("Preview", cv.WINDOW_NORMAL)
cv.namedWindow("Preview2", cv.WINDOW_NORMAL)
# Initialize a slicer
slicer = dv.EventStreamSlicer()

# Declare the callback method for slicer
def slicing_callback(events: dv.EventStore):
    # Pass events into the accumulator and generate a preview frame
    accumulator.accept(events)
    frame = accumulator.generateFrame()

    # print(frame.image.shape)
    # print(np.max(frame.image))
    # print(np.min(frame.image))

    processed_frame, result = process_image(frame.image, 45)

    print(np.min(processed_frame))
    print(np.max(processed_frame))

    # Show the accumulated image
    cv.imshow("Preview", result[:, :, 0])
    cv.imshow("Preview2", result[:, :, 1])

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