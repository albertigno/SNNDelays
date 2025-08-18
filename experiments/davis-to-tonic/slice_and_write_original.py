'''
demonstration of slicing events with a predefined period, and saving them one by one
'''

import dv_processing as dv
import datetime
import cv2 as cv
import os

# basic configuration
## folder
location = 'A'
if not os.path.exists(location):
    os.makedirs(location)

file_name = 'sample'
time_ms = 250
#num_samples = 720
num_samples = 5

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

# Create the preview window
cv.namedWindow("Preview", cv.WINDOW_NORMAL)

def preview_events(event_slice):
    cv.imshow("Preview", visualizer.generateImage(event_slice))
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
    