import cv2
import io
import json
import numpy as np
import os
import pandas as pd
import prepare_image_data as pid
import plotly.express as px
import re


import os
import PIL
import torch
import torchvision.transforms as T
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from time import time, time_ns
import torch
import torch.nn.functional as F

import torchmetrics
import torchvision.datasets as datasets
import yaml

from ast import literal_eval
from datetime import datetime
from IPython.display import display
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
from time import time
from torch.autograd import Variable
from torchmetrics.classification import MulticlassAccuracy
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch import tensor


class CellImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.classes = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.classes.iloc[idx, 0])
        image = cv2.imread(img_name)
        image = Image.fromarray(np.uint8(image))
        label = self.classes.iloc[idx, 1:]
        label = np.array([label])
        label = torch.tensor(label.reshape(-1).tolist())
        if self.transform:
            image = self.transform(image)
            image = torch.Tensor.double(image)
        return image, label

class CNN(torch.nn.Module):
    
    """A neural network building class.
    Args: 
       kernel: A list of integers reflecting the kernel sizes
    Returns:
        Neural network architecture to be trained
    """
    def __init__(self, kernel):
        super().__init__()
        #define layers
        self.layers = torch.nn.Sequential(
            #changed to 1d as not an image. Num channels, num outputs
            torch.nn.Conv2d(1, 4, kernel[0]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 8, kernel[1]),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(30, 30),
            #This is important.
            torch.nn.Flatten(),
            # LazyLinear removes need for manual calculations of input from the flattened layer
            torch.nn.LazyLinear(2),
            torch.nn.Softmax()
        )
    #define how to stack the different elements of the network together
    def forward(self, X):
            return self.layers(X)

def train_cell_image_nn(model, num_epochs:int, name_writer:str, dataloader, lr, optimiser):
    """Trains a neural network model
    Args:  
        model: A neural nework model instance
        num_epochs: The number of epochs to train the model on (int)
        name_writer: A name for Tensorboard graph (string)
        dataloader: A dataloader instance for feeding the model
        lr: The learning rate for an instance of the model
        optimiser: The optimiser to use for the training process
    Returns:
        training_metrics: A dictionary containing training_duration and training_loss}
        model_parameters: A dictionary containing optimiser_parameters, model state dictionary and batch size"""

    #The algorithm for this optimisation step is stochastic gradient descent (since linear regression is used to derive the model). Passing model.parameters and the learning rate lr
    optimiser = optimiser(model.parameters(), lr=lr)
    writer = SummaryWriter()
    #changing parameters to double type
    model.double()
    batch_idx = 0 
    # created this as a counter for writer since the batch resets within the loop
    start_time = time()
    training_loss_list = []
    training_acc_list = []
    for epoch in range(num_epochs):
        #iterates in batches for the number of epochs defined
        for batch in dataloader:
            print(batch)
            #unpack the batch into features and labels
            features, label = batch
            #make a prediction from the model based on a batch of features
            prediction = model(features)
            print(prediction)
            label = torch.Tensor.double(label)
            #calculate the loss- used to apply to gradient descent algorithm. Using cross entropy
            training_loss = F.binary_cross_entropy(prediction, label)
            print(f'training_loss: {training_loss}')
            training_acc = multiclass_accuracy(prediction, label)
            # prediction_flattened = torch.flatten(prediction)
            training_acc_list.append(training_acc)
            # #populates gradients of model parameters with respect to loss
            diff = training_loss.backward()
            #opimisation step 
            optimiser.step()
            # #the .grad associated with tensor .backward does not go back to 0 with every iteration (clearly this would cause issues with SGD), accordingly must re-zero the grad of the optimiser after each iteration. But don't do this before .step!
            optimiser.zero_grad()
            #Adding scalar to Tensorboard
            writer.add_scalar(name_writer, training_loss.item(), batch_idx)
            batch_idx += 1
            #Need to detach here to calculate mean and slice list
            training_loss = training_loss.detach().numpy()
            training_loss_list.append(training_loss)
            #truncates the loss list to last 10, with which to get average
            if len(training_loss_list) > 30:
                training_loss_list = training_loss_list[-29:]
            if len(training_acc_list) > 30:
                training_acc_list = training_acc_list[-29:]
    #calculate the mean entropy loss from the last 30 iterations
    training_loss_mean = np.mean(training_loss_list)
    training_acc_mean = np.mean(training_acc_list)
    end_time = time()
    training_duration = end_time - start_time
    optimiser_parameters = optimiser.state_dict()
    model_state_dict = model.state_dict()
    model_parameters = model_state_dict | optimiser_parameters
    training_metrics = {'training_duration': training_duration, 'loss_mean':training_loss_mean, 'training_acc_mean': training_acc_mean}
    return training_metrics, model_parameters

def multiclass_accuracy(prediction, labels): 
    assert len(labels) == len(prediction)
    n_correct = 0; n_wrong = 0 
    pred_idx = 0; labels_idx = 0
    for i, single_prediction in enumerate(prediction):
        #print(f'single_prediction: {single_prediction}')
        pred_idx = torch.argmax(single_prediction)
        #print(f'pred_idx = {pred_idx}') 
        labels_idx = torch.argmax(labels[i]) 
        #print(f'labels_idx = {labels_idx}')
        #print(labels_idx)
        if pred_idx == labels_idx:
            n_correct += 1
        else:
            n_wrong += 1
        #print(f'n_wrong = {n_wrong}')
    print(f'n_correct: {n_correct}')
    acc = (n_correct * 1.0) / (n_correct + n_wrong)
    return acc

def evaluate_model(model, dataloader_val, dataloader_test=None):
    """Evaluates neural network performance
    Args:
        model: neural network model instance
        dataloader_val: Dataloder instance for the validation dataset
        dataloader_test(optional): Dataloder instance for the test dataset
    Returns: Dictionary of performance metrics including: validation loss (mse), validation r2, test loss (mse), test r2, mean inference latency(ms)
        """
    model.eval()
    #we don't need the grad function here so switching it off with no_grad
    val_loss_list = []
    test_loss_list = []
    inference_latencies_list = []
    val_acc_list = []
    test_acc_list = []
    val_acc_mean = 0
    test_acc_mean = 0
    with torch.no_grad():
        for x, y in dataloader_val:
            #starting the timer for inference latency
            inference_latency_start = time_ns()
            #making the prediction
            prediction = model(x)
            y = torch.Tensor.double(y)
            #appending the latency into a list
            inference_latencies_list.append(time_ns() - inference_latency_start)
            val_loss = F.binary_cross_entropy(prediction, y)
            print(f'val_loss: {val_loss}')
            val_acc = multiclass_accuracy(prediction, y)
            val_acc_list.append(val_acc)
            print(f'val_acc: {val_acc}')
            val_loss = val_loss.detach().numpy()
            val_loss_list.append(val_loss)
    val_loss_mean = np.mean(val_loss_list)
    print(val_loss_mean)
    val_acc_mean = np.mean(val_acc_list)
    print(f'val_acc_mean: {val_acc_mean}')
    if dataloader_test == None:
        test_loss, test_acc_mean = 'n/a', 'n/a'
           
    #we don't need the grad function here so switching it off with no_grad
    else: 
        with torch.no_grad():
            for x, y in dataloader_test:
                inference_latency_start = time()
                prediction = model(x)
                inference_latencies_list.append(time() - inference_latency_start)
                test_loss = F.mse_loss(prediction, y)
                print(f'test_loss: {test_loss}')
                #need to instantiate r2 for this to work
                test_acc = multiclass_accuracy(y, y)
                test_acc_list.append(test_acc)
                #truncates the loss list to last 30, with which to get more reliable average
                if len(test_loss_list) > 30:
                    test_loss_list = test_loss_list[-29:]
    test_loss_mean = np.mean(test_loss_list)
    print(test_loss_mean)
    test_acc_mean = np.mean(test_acc_list)
    print(f'test_acc_mean: {test_acc_mean}')

    mean_inference_latency = np.mean(inference_latencies_list)
    
    performance_metrics = {'val_loss_mean':val_loss_mean, 'val_acc_mean':val_acc_mean, 'test_loss_mean':test_loss_mean, 'test_acc_mean':test_acc_mean, 'mean_inference_latency':mean_inference_latency}
    return performance_metrics

if __name__ == '__main__':
    transforms = T.Compose([
                        T.Resize([62, 156]),
                        T.Grayscale(),
                        T.ToTensor(),
                        T.Normalize((0.5, ), (0.5, ))
                        ])
    training_dataset = CellImageDataset(r'C:\Users\marko\DS Projects\Machine-learning-with-biological-data\Main_project_file\train_dataset.csv', r'C:\Users\marko\DS Projects\Machine-learning-with-biological-data\Main_project_file\image datasets\train_images', transform=transforms)
    for i in range(len(training_dataset)):
        image, label  = training_dataset[i]
        print(i, image.size(), label)
    batch_size = 10
    dataloader_train = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
    NN_inst = CNN([8, 8])
    n_iters = len(dataloader_train) * 100
    num_epochs = pid.get_num_epochs(n_iters, batch_size, training_dataset)
    training_metrics, model_parameters = train_cell_image_nn(NN_inst, num_epochs, 'cell images first pass', dataloader_train, 0.1, torch.optim.SGD)
    val_dataset = CellImageDataset(r'C:\Users\marko\DS Projects\Machine-learning-with-biological-data\Main_project_file\val_dataset.csv', r'C:\Users\marko\DS Projects\Machine-learning-with-biological-data\Main_project_file\image datasets\val_images', transform=transforms)
    dataloader_val = DataLoader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=True)
    eval_metrics = evaluate_model(NN_inst, dataloader_val, dataloader_test=None)    



    

