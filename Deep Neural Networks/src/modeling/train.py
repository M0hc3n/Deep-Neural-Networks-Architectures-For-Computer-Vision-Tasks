import numpy as np
import matplotlib.pyplot as plt

import matplotlib.ticker as mticker

from modeling.metrics import accuracy, test_loss

from core.config import report_dir
class ModelTrainer:
    training_data_loader = None
    testing_data_loader = None

    model, criterion, optimizer = None, None, None
    
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    epochs = 5

    def __init__(self, training_data_loader, testing_data_loader, model, criterion, optimizer, epochs):
        self.training_data_loader = training_data_loader
        self.testing_data_loader = testing_data_loader

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.epochs = epochs

    def trainer_each_batch(self, x, y):
        self.model.train()
        # call model on the batch of inputs
        # `model.train()` tells your model that you are training the model. 
        # So BatchNorm layers use per-batch statistics and Dropout layers are activated etc
        # Forward pass: Compute predicted y by passing x to the model
        prediction = self.model(x)[0]
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

        for epoch in range(self.epochs):
            print(epoch)
            
            # Iteration to calculate train_losses
            # Creating lists that will contain the accuracy and loss values corresponding to each batch within an epoch:
            losses_in_this_epoch, train_accuracies_in_this_epoch = [], []
            
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
            train_epoch_loss = np.array(losses_in_this_epoch).mean()
            
            # Iteration to calculate train_accuracies
            # Next, we calculate the accuracy of the prediction at the end of training on all batches:
            for ix, batch in enumerate(iter(self.training_data_loader)):
                x, y = batch
                is_correct = accuracy(x, y, self.model)
                train_accuracies_in_this_epoch.extend(is_correct)
            epoch_accuracy = np.mean(train_accuracies_in_this_epoch)
            
            """ Calculate the loss value and accuracy within the one batch of test data
            Note, that the batch size of the test data is equal to the length of the entire test data. 
            So its just a single batch to cover the entire test data.
            """
            for ix, batch in enumerate(iter(self.testing_data_loader)):
                x, y = batch
                test_is_correct = accuracy(x, y, self.model)
                test_epoch_loss = test_loss(x, y, self.model, self.criterion)
            test_epoch_accuracy = np.mean(test_is_correct)
            
            self.train_losses.append(train_epoch_loss)
            self.train_accuracies.append(epoch_accuracy)
            
            self.test_losses.append(test_epoch_loss)
            self.test_accuracies.append(test_epoch_accuracy)
    
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

    def plot_model_weights(self):
        for ix, par in enumerate(self.model.parameters()):
            if(ix==0):
                plt.hist(par.cpu().detach().numpy().flatten())
                plt.title("Histogram for weights - connecting the input layer to the hidden layer")

                plt.savefig(f"{report_dir}/Histogram for weights - connecting the input layer to the hidden layer.png")

            if(ix==1):
                plt.hist(par.cpu().detach().numpy().flatten())
                plt.title("Histogram - Bias in the hidden layer")

                plt.savefig(f"{report_dir}/Histogram - Bias in the hidden layer.png")

            if(ix==2):
                plt.hist(par.cpu().detach().numpy().flatten())
                plt.title("Histogram for weights - connecting the hidden layer to the output layer")

                plt.savefig(f"{report_dir}/Histogram for weights - connecting the hidden layer to the output layer.png")

            if(ix==3):
                plt.hist(par.cpu().detach().numpy().flatten())
                plt.title("Histogram - Bias in the output layer")

                plt.savefig(f"{report_dir}/Histogram - Bias in the output layer.png")


    

