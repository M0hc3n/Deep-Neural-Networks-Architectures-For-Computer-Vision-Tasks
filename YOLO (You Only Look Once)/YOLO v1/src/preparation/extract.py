from core.loggging import logger

from torch.utils.data import Dataset
from torch import zeros, tensor

import pandas as pd

from PIL import Image
import os

class VOCDatasetExtractor(Dataset):
    def __init__(self, csv_file, image_dir, label_dir, S=7, B=2, C=20, transform=None):
        """
        Initialize the VOCDatasetExtractor.

        Args:
            csv_file (str): Path to the CSV file containing annotations.
            image_dir (str): Directory containing the images.
            label_dir (str): Directory containing the label files.
            S (int, optional): Grid size. Defaults to 7.
            B (int, optional): Number of bounding boxes per grid cell. Defaults to 2.
            C (int, optional): Number of classes. Defaults to 20.
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
        """
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path, "r") as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x) for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height]) # we use this as our format for each element

        image_path = os.path.join(self.image_dir, self.annotations.iloc[index, 0])
        image = Image.open(image_path)
        boxes = tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        # label matrix is formatted as follows:
        # [:, :, 0:19] -> 20 values for each class (one hot encoded)
        # [:, :, 20] -> confidence score for each bounding box
        # [:, :, 21:25] -> class probabilities for each bounding box
        label_matrix = zeros((self.S, self.S, self.C + 5 * self.B))

        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # we need to convert x,y,width,height into grid cell coordinates
            # we need to find the cell that contains the object
            i, j = int(y * self.S), int(x * self.S)
            
            cell_x, cell_y = self.S * x - j, self.S * y - i #! I need to check this afterwards


            width_cell, height_cell = width * self.S, height * self.S

            if label_matrix[i, j, 20] == 0: # no object found in this cell
                label_matrix[i, j, 20] = 1 # we found an object

                label_matrix[i, j, 21:25] = tensor([cell_x, cell_y, width_cell, height_cell])

                label_matrix[i, j, class_label] = 1 # we found the object

        return image, label_matrix
    


