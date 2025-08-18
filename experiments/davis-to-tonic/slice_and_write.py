'''
demonstration of slicing events with a predefined period, and saving them one by one
'''

import dv_processing as dv
import datetime
import cv2 as cv
import os
import torch
import tonic.transforms as transforms
import numpy as np

# basic configuration
## folder
folder_name = 'O_test'
location = os.path.join(r"E:\SNN_DATASETS\Datasets", folder_name)
if not os.path.exists(location):
    os.makedirs(location)
file_name = 'sample'

# # ABCXO configuration
# time_ms = 250
# num_samples = 720 # 3 minutes (600 train, 120 test)

time_ms = 50
#num_samples = 2000 # 100 secs train
num_samples = 1000 # 50 secs test

tonic_size = 32

# Open the camera
camera = dv.io.CameraCapture("DAVIS240C_02460013")

resolution = camera.getEventResolution()
print("Camera resolution: ", resolution)

# Initialize visualizer instance which generates event data preview
visualizer = dv.visualization.EventVisualizer(resolution)

# global counter for the write slicer
x=0

# Event only configuration for the writer
config = dv.io.MonoCameraWriter.EventOnlyConfig("DAVIS240C_02460013", resolution)

# tonic preview
tonic_dtype = np.dtype([("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)])

sensor_size = (240, 180, 2)
transforms_list = []

transforms_list.append(
    transforms.CenterCrop(sensor_size=sensor_size, size=(128, 128)))
cropped_sensor_size = (128, 128, 2)

target_size = (tonic_size, tonic_size)
spatial_factor = \
    np.asarray(target_size) / cropped_sensor_size[:-1]

transforms_list.append(
    transforms.Downsample(spatial_factor=spatial_factor[0]))

transforms_list.append(
    transforms.ToFrame(
        sensor_size=(tonic_size, tonic_size, 2), n_time_bins=1))

to_frame = transforms.Compose(transforms_list)

# Create the preview window
cv.namedWindow("Preview", cv.WINDOW_NORMAL)
cv.resizeWindow("Preview", 500, 500)

def preview_events(events: dv.EventStore):
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

        frame = to_frame(tonic_events)

        if len(frame.shape)!=4:
            frame = np.zeros((1, 2, tonic_size, tonic_size)).astype(np.int16)

        print(np.max(frame))

        composed_frame = 255*frame[0, 1, :, :] - 255*frame[0, 0, :, :]
        max = np.max(np.abs(composed_frame))

        composed_frame = (1.0 + composed_frame / max) / 2.0

        cv.imshow("Preview", composed_frame)
        cv.waitKey(2)

def write_events(event_slice):
    global x
    # Create the writer instance, it will only have a single event output stream.
    writer = dv.io.MonoCameraWriter(os.path.join(location, f"{file_name}_{x:04d}.aedat4"), config)
    writer.writeEvents(event_slice)
    x += 1

# Create an event visualization slicer
slicer = dv.EventStreamSlicer()
slicer.doEveryTimeInterval(datetime.timedelta(milliseconds=20), preview_events)

# Create the write event slicer (every second)
write_slicer = dv.EventStreamSlicer()
write_slicer.doEveryTimeInterval(datetime.timedelta(milliseconds=time_ms), write_events)

# start read loop
while x<num_samples:

    # Get events
    events = camera.getNextEventBatch()

    # If no events arrived yet, continue reading
    if events is not None:
        slicer.accept(events)
        write_slicer.accept(events)
        
        



# # Write 100 packet of event data
# for i in range(100):
#     # EventStore requires strictly monotonically increasing data, generate
#     # a timestamp from the iteration counter value
#     timestamp = i * 1000

#     # Empty event store
#     events = dv.data.generate.dvLogoAsEvents(timestamp, resolution)

#     # Write the packet using the writer, the data is not going be written at the exact
#     # time of the call to this function, it is only guaranteed to be written after
#     # the writer instance is destroyed (destructor has completed)
    