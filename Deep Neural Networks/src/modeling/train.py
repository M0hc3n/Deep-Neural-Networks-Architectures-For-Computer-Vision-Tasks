import numpy as np
import matplotlib.pyplot as plt

from modeling.metrics import accuracy

from core.config import report_dir
class ModelTrainer:
    training_data_loader = None
    model, criterion, optimizer = None, None, None
    losses, accuracies = [], []

    def __init__(self, training_data_loader, model, criterion, optimizer):
        self.training_data_loader = training_data_loader

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer


    def trainer_each_batch(self, x, y):
        self.model.train()
        # call model on the batch of inputs
        # `model.train()` tells your model that you are training the model. 
        # So BatchNorm layers use per-batch statistics and Dropout layers are activated etc
        # Forward pass: Compute predicted y by passing x to the model
        prediction = self.model(x) 
        # "prediction = model(x)" executes forward propagation over x 
        
        # compute loss
        loss_for_this_batch = self.criterion(prediction, y)
        
        # based on the forward pass in `model(x)` compute all the gradients of model.parameters()
        # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True.
        # These are accumulated into x.grad for every parameter x.
        # In pseudo-code:
        # x.grad += dloss/dx
        loss_for_this_batch.backward()
        
        """ apply new-weights = f(old-weights, old-weight-gradients) where "f" is the optimizer 
        When you call `loss.backward()`, all it does is compute gradient of loss w.r.t all the parameters in loss that have `requires_grad = True` and store them in `parameter.grad` attribute for every parameter.

        `optimizer.step()` updates all the parameters based on `parameter.grad`
        """
        
        self.optimizer.step()
        
        # Flush gradients memory for next batch of calculations
        # `optimizer.zero_grad()` clears `x.grad` for every parameter x in the optimizer.
        # Not zeroing grads would lead to gradient accumulation across batches.
        self.optimizer.zero_grad()
        
        return loss_for_this_batch.item()

    def train_model(self):
        for epoch in range(5):
            print(epoch)
            
            # Creating lists that will contain the accuracy and loss values corresponding to each batch within an epoch:
            losses_in_this_epoch, accuracies_in_this_epoch = [], []
            
            # Create batches of training data by iterating through the DataLoader:
            for ix, batch in enumerate(iter(self.training_data_loader)):
                x, y = batch
                """ Train the batch using the trainer_each_batch() function and store the loss value at
                the end of training on top of the batch as loss_for_this_batch. 
                Furthermore, store the loss values across batches in the losses_in_this_epoch list:
                """
                loss_for_this_batch = self.trainer_each_batch(x, y)
                losses_in_this_epoch.append(loss_for_this_batch)
            
            # After the above loop is done 
            # store the mean loss value across all batches within an epoch:    
            epoch_loss = np.array(losses_in_this_epoch).mean()
            
            # Next, we calculate the accuracy of the prediction at the end of training on all batches:
            for ix, batch in enumerate(iter(self.training_data_loader)):
                x, y = batch
                is_correct = accuracy(x, y, self.model)
                accuracies_in_this_epoch.extend(is_correct)
            epoch_accuracy = np.mean(accuracies_in_this_epoch)
            
            self.losses.append(epoch_loss)
            
            self.accuracies.append(epoch_accuracy)
    
    def plot_trainning_report(self):
        epochs = np.arange(5)+1

        print(self.losses)
        print(self.accuracies)
        plt.figure(figsize=(20,5))
        plt.subplot(121)
        plt.title('Loss value over increasing epochs')
        plt.plot(epochs, self.losses, label='Training Loss')
        plt.legend()
        plt.subplot(122)
        plt.title('Accuracy value over increasing epochs')
        plt.plot(epochs, self.accuracies, label='Training Accuracy')
        plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()]) 
        plt.legend()

        plt.savefig(f"{report_dir}/classification_report.png")

