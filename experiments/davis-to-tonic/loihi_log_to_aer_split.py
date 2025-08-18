import dv_processing as dv
import numpy as np
import os
# Sample VGA resolution, same as the DVXplorer camera
#resolution = (32, 32)
resolution = (64, 64)

# Event only configuration
#config = dv.io.MonoCameraWriter.EventOnlyConfig("DVXplorer_sample", resolution)

folder_name = 'spikesO'

location = os.path.join(r"E:\SNN_DATASETS\Datasets", folder_name)

log_file_location = os.path.join(r"P:\NewLoihiData\Loihi_dataset\ABCXO_64_robust", folder_name+".log")

split_time_ms = 100

if not os.path.exists(location):
    os.makedirs(location)

def get_writer(file_name):

    config = dv.io.MonoCameraWriter.Config("DVXplorer_sample")
    # Add an event stream with a resolution
    config.addEventStream(resolution)
    # Add frame stream with a resolution
    config.addFrameStream(resolution)
    # Add IMU stream
    config.addImuStream()
    # Add trigger stream
    config.addTriggerStream()

    # Create the writer instance, it will only have a single event output stream.
    writer = dv.io.MonoCameraWriter(file_name, config)

    return writer

# Read text data line by line and parse events
events_store = dv.EventStore()

c = 0
timestamps = []

# dummy frame
image = np.full((resolution[0], resolution[1], 3), fill_value=255, dtype=np.uint8)

with open(log_file_location, "r") as f:
#with open("spikes_output_loihi.log", "r") as f:
    for idx, line in enumerate(f):
        timestamp_str, x_str, y_str, polarity_str = line.strip().split(",")
        timestamp = int(float(timestamp_str) * 1_000_000)  # Convert seconds to microseconds
        if idx==0:
            first_timestamp = timestamp
        x = int(x_str)
        y = resolution[0]-1-int(y_str)
        polarity = bool(int(polarity_str))
        events_store.push_back(timestamp, x, y, polarity)

        if timestamp-first_timestamp>split_time_ms*1000:
            file_name = os.path.join(location, f"sample_{c:04d}.aedat4")

            writer = get_writer(file_name)

            first_timestamp = timestamp
            timestamps.append(timestamp)
            frame = dv.Frame(timestamp, image)
            c += 1

            writer.writeEvents(events_store)
            writer.writeFrame(frame)

            # empty events store
            events_store = dv.EventStore()

            #break



### then run aedat4to2 events_from_loihi.aedat4 --no_imu --no_frame   