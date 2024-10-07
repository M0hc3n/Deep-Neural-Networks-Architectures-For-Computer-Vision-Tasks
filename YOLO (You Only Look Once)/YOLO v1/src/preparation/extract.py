from core.loggging import logger

from torch.utils.data import Dataset
from torch import zeros

import pandas as pd

class VOCDatasetExtractor(Dataset):
    def __init__(self, csv_file, image_dir, label_dir, S=7, B=2, C=20, transform=None):
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
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        # now, we convert to cells
        label_matrix = zeros((self.S, self.S, self.C + 5 * self.B))


    
    


