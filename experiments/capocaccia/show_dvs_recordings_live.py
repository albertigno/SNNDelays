import torch
import time
import cv2 as cv
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.train_utils_refact_minimal import get_device
import numpy as np
import time
from snn_delays.config import DATASET_PATH
import os
'''
SHD dataset as in ablation study
'''

device = get_device()
#dataset = 'ibm_gestures'
#dataset = 'SOX_01.npy'
#dataset = 'xiao_hong_shu.npy'
#dataset = 'C.npy'
dataset = 'davis_frames_X.npy'

#dataset = os.path.join(DATASET_PATH, 'Capocaccia', 'randomcaccia.npy')


if dataset == 'ibm_gestures':

    num_steps = 50
    batch_size = 1
    # DATASET
    DL = DatasetLoader(dataset=dataset,
                    caching='',
                    num_workers=0,
                    batch_size=batch_size,
                    total_time=num_steps,
                    sensor_size_to=64,
                    crop_to=1e6)
    train_loader, test_loader, dataset_dict = DL.get_dataloaders()

else:
    test_loader = np.load(dataset)
    #test_loader = np.load(dataset)[:1000]
    print(test_loader.shape)

cv.namedWindow("Preview", cv.WINDOW_NORMAL)

def on_trackbar(val):
    global slider_value
    slider_value = val

slider_value = 0
cv.createTrackbar("Frame", "Preview", 0, len(test_loader) - 1, on_trackbar)

while True:
    
    #slider_value += 1    
    
    im_step = test_loader[slider_value]

    # frame = im_step[0, 0, :, :].astype(np.float)/255.0
    # #frame = im_step[0, 0, :, :].astype(np.float)

    #frame = im_step
    frame = 255*im_step[0, 0, :, :].astype(np.uint8)

    print(frame.shape)

    print(np.max(frame))
    print(np.min(frame))

    cv.putText(frame, f"Frame: {slider_value}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv.imshow("Preview", frame)
    #cv.putText(frame, f"Frame: {slider_value}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    key = cv.waitKey(1)

    if key & 0xFF == ord('q'):  # Press 'q' to quit
        break







# cv.namedWindow("Preview", cv.WINDOW_NORMAL)

# for images, labels in test_loader:

#     for step in range(num_steps):
#         im_step = images[:, step, :, :, :]

#         frame = im_step[0, 0, :, :].detach().cpu().numpy()

#         print(np.max(frame))
#         print(np.min(frame))

#         time.sleep(0.1)

#         cv.imshow("Preview", frame/60)

#         key = cv.waitKey(1)

#         if key & 0xFF == ord('p'):
#             while True:
#                 key = cv.waitKey(0)
#                 if key == ord('p'):
#                     break