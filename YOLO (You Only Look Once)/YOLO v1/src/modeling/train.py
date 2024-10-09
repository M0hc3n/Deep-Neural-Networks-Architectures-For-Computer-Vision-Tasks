import numpy as np
import matplotlib.pyplot as plt

import matplotlib.ticker as mticker

from modeling.metrics import mean_average_precision
from preparation.utils import get_bboxes

from core.config import report_dir, device

from tqdm import tqdm

class ModelTrainer:
    training_data_loader = None
    testing_data_loader = None

    model, criterion, optimizer = None, None, None
    
    mean_loss = []

    epochs = 5

    def __init__(self, training_data_loader, testing_data_loader, model, criterion, optimizer, epochs):
        self.training_data_loader = training_data_loader
        self.testing_data_loader = testing_data_loader

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.epochs = epochs

    def trainer_each_epoch(self):
        looper = tqdm(self.training_data_loader, leave=True)
        batch_loss = []
        
        for idx, (x, y) in enumerate(looper):
            x, y = x.to(device), y.to(device)
            out = self.model(x)
            loss = self.criterion(out, y)
            
            batch_loss.append(loss.item())


            self.optimizer.zero_grad()
            los.backward()
            self.optimizer.step()
            
            looper.set_postfix(loss=loss.item())
            
        return sum(batch_loss) / len(batch_loss)

    def train_model(self):
        
        for epoch in range(self.epochs):
            pred_boxes, target_boxes = get_bboxes(self.training_data_loader, self.model, iou_threshold=0.5, threshold=0.4)
            
            mean_avg_precision = mean_average_precision(
                pred=pred_boxes, target=target_boxes, iou_threshold=0.5
            )
            
            print(f"EPOCH: {epoch}.      Mean Average Precision Value: {mean_avg_precision}")
            
            self.trainer_each_epoch()

    def plot_trainning_report(self):
                
        epochs = np.arange(self.epochs)+1

        plt.figure(figsize=(20,5))

        plt.subplot(121)

        plt.plot(epochs, self.train_losses, 'blue', label='Training Losses')
        plt.plot(epochs, self.test_losses, 'r', label='Test Losses')

        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.title('Training and Test Losses vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.legend()
        plt.grid('off')
        plt.show()

        plt.subplot(122)

        plt.plot(epochs, self.train_accuracies, 'blue', label='Training Accuracies')
        plt.plot(epochs, self.test_accuracies, 'r', label='Test Accuracies')

        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

        plt.title('Training and Test Accuracies vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracies')
        plt.legend()
        plt.grid('off')
        plt.savefig(f"{report_dir}/classification_report.png")


