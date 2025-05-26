import os
import dv_processing as dv

from typing import Callable, Optional

import numpy as np

from tonic.dataset import Dataset
from tonic_datasets import TonicDataset
from snn_delays.config import DATASET_PATH

class DAVIS240C(Dataset):
    """
    Davis240c (Alberto's implementation)
    """

    # test_url = "https://figshare.com/ndownloader/files/38020584"
    # train_url = "https://figshare.com/ndownloader/files/38022171"
    # test_md5 = "56070e45dadaa85fff82e0fbfbc06de5"
    # train_md5 = "3a8f0d4120a166bac7591f77409cb105"
    # test_filename = "ibmGestureTest.tar.gz"
    # train_filename = "ibmGestureTrain.tar.gz"

    sensor_size = (240, 180, 2)
    dtype = np.dtype([("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)])
    ordering = dtype.names

    def __init__(
        self,
        parent_dir: str,
        save_to: str,
        # train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(
            save_to,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

        # self.train = train

        # if train:
        #     self.url = self.train_url
        #     self.file_md5 = self.train_md5
        #     self.filename = self.train_filename
        #     self.folder_name = "ibmGestureTrain"
        # else:
        #     self.url = self.test_url
        #     self.file_md5 = self.test_md5
        #     self.filename = self.test_filename
        #     self.folder_name = "ibmGestureTest"

        # if not self._check_exists():
        #     self.download()

        self.classes = sorted(entry for entry in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, entry)))

        self.data = []
        self.target = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(parent_dir, class_name)
            for sample_name in os.listdir(class_dir):
                sample_path = os.path.join(class_dir, sample_name)
                if os.path.isfile(sample_path):  # ensure it's a file
                    self.data.append(sample_path)
                    self.target.append(label)

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """

        # Open davis aedat4 file
        #reader = dv.io.MonoCameraRecording(r"C:\Users\Alberto\Python\SNNDelays\some_category\sample_0000.aedat4")
        reader = dv.io.MonoCameraRecording(self.data[index])
        events_packets = []
        while reader.isRunning():
            events = reader.getNextEventBatch()
            if events is not None:
                events_packets.append(events.numpy())
        events_packets = np.concatenate(events_packets)

        # Extract fields and convert
        x = events_packets['x']
        y = events_packets['y']
        p = events_packets['polarity'].astype(bool)  # Convert to bool
        t = events_packets['timestamp']

        events = np.empty(len(events_packets), dtype=self.dtype)
        events["x"] = x
        events["y"] = y
        events["p"] = p
        events["t"] = t

        target = self.target[index]
        #target = 0

        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            events, target = self.transforms(events, target)

        return events, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return (
            self._is_file_present()
            and self._folder_contains_at_least_n_files_of_type(100, ".npy")
        )


class DAVIS240Dataset(TonicDataset):
    """
    DAVIS240
    the expcted file structure is:
    
    -train
        - class 1
            -sample 1
            -sample 2
            -etc  
        - class 2
        - etc
    test
        - class 1
            -sample 1
            -sample 2
            -etc  
        - class 2
        - etc

    """

    def __init__(self, dataset_name='davis', total_time=50, **kwargs):
        super().__init__(dataset_name=dataset_name,
                         total_time=total_time,
                         **kwargs)

        path = os.path.join(DATASET_PATH, kwargs['folder_name'])
        test_path = os.path.join(path, 'test')
        train_path = os.path.join(path, 'train')

        self.n_classes = self.classes = len([entry for entry in os.listdir(train_path)
                                             if os.path.isdir(os.path.join(train_path, entry))])
        self.set_target_transform()

        # Train and test dataset definition
        self.train_dataset = DAVIS240C(
            save_to='',
            parent_dir=train_path,
            transform=self.sample_transform,
            target_transform=self.label_transform)

        self.test_dataset = DAVIS240C(
            save_to='',
            parent_dir=test_path,
            transform=self.sample_transform,
            target_transform=self.label_transform)