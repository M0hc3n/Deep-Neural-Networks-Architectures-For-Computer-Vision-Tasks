from torch.utils.data import Dataset

from core.config import device

class DatasetTransformer(Dataset):
    """ Will take 2 arrays/tensors for parameters train_images (x) and train_img_targets (y) """

    def __init__(self, x, y):
        # converting the input into a floating-point number
        # And also scale them to the range of [0,1] by dividing by 255
        x = x.float() / 255

        #  flattened each image into 28*28 = 784 numeric values
        # where each numeric value corresponds to a pixel value
        x = x.view(-1, 28 * 28)
        self.x, self.y = x, y

    # __getitem__ function returns a sample from the dataset given an index.
    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        # It is necessary to have both the model, and the data on the same device,
        # either CPU or GPU, for the model to process data.
        # Data on CPU and model on GPU, or vice-versa, will result in a Runtime error.
        # to(device) => this method will move the tensor or model to the specified device.
        return x.to(device), y.to(device)

    # __len__ function which returns the size of the dataset,
    def __len__(self):
        return len(self.x)
