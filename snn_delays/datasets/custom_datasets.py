
from tonic.dataset import Dataset

class CustomDataset(Dataset):
    """
    Dataloader for custom numpy or pytorch dataset.
    """
    def __init__(self, data, labels):
        """
        Initialization of the class.

        :param data: Input data.
        :param labels: Labels of the input data.
        """

        assert len(data)==len(labels), \
            "[ERROR] Data length must be equal to labels length."

        # Set attributes from input
        self.images = data
        # shape (num_samples, num_timesteps, num_input_neurons)
        self.labels = labels
        # shape (num_samples, num_output_neurons)

    def __len__(self):
        """
        The number of samples in the dataset.

        :return: Dataset size.
        """
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        """
        Get a sample of the dataset.

        :param idx: Index of the sample to be returned.
        :return: A tuple with the original (sample) and the target (label)
        sequence.
        """

        img, target = self.images[idx], self.labels[idx]
        return img, target    
    
    def get_train_attributes(self):
        """
        Function to get these three attributes which are necessary for a
        correct initialization of the SNNs: num_training samples, num_input...
        All Dataset should have this, if possible.
        """
        train_attrs = {'num_input': self.images.shape[2],
                       'num_training_samples': len(self),
                       'num_output': self.labels.shape[1]}

        return train_attrs        
