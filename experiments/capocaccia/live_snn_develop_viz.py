import cv2
import numpy as np
from collections import deque

from cool_vizualizer_tools import draw_mems, draw_spks

from snn_delays.utils.model_loader_refac import ModelLoader

snn = ModelLoader('ibm_gest_ffw', 'capocaccia_live', 1, 'cpu', live = True)

import torch
import time
from snn_delays.utils.dataset_loader import DatasetLoader
from snn_delays.utils.train_utils_refact_minimal import get_device

'''
SHD dataset as in ablation study
'''

device = get_device()
dataset = 'ibm_gestures'
num_steps = 50
batch_size = 1

# DATASET
DL = DatasetLoader(dataset=dataset,
                  caching='',
                  num_workers=0,
                  batch_size=batch_size,
                  total_time=num_steps,
                  sensor_size_to=32,
                  crop_to=1e6)
train_loader, test_loader, dataset_dict = DL.get_dataloaders()

# # Constants
# WIDTH, HEIGHT = 800, 600  # Canvas dimensions
# NEURON_COUNT = 11         # Number of neurons (from your data)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
          (255, 0, 255), (0, 255, 255), (255, 128, 0), 
          (128, 0, 255), (0, 255, 128), (128, 255, 0), (255, 0, 128)]  # Unique colors

# # Initialize plot history (stores last N timesteps)
# max_history = WIDTH // 2  # Adjust based on desired horizontal resolution
# history = deque(maxlen=max_history)

# # Create a blank canvas
# canvas = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255  # White background

for images, labels in test_loader:

    for step in range(num_steps):
        im_step = images[:, step, :, :, :]
        print(im_step.shape)
        pred = snn.propagate_live(im_step)    

        #draw_mems(snn.mems_fifo['output'], COLORS)
        draw_spks(snn.spikes_fifo['l1'], 64)

    snn.reset_state_live()

    print(f'pred: {pred}, ref: {labels}')


cv2.destroyAllWindows()





