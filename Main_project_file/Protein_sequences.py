import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import re
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets

from ast import literal_eval
from datetime import datetime
from IPython.display import display
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from time import time, time_ns
from torch import tensor
from torch.autograd import Variable
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics import ConfusionMatrix
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter


pd.options.mode.chained_assignment = None

def prep_protein_seq_data(path_to_csv, save_data=False, path_to_new_data=False):
    '''Prepares amino acid data from a .csv file. Preparation involves excluding amino acids of 
    length less than 50 amino acid base pairs, while also excluding sequences containing 'x'. A
    new .csv file with the cleaned data is saved.
    Args:
        path_to_csv: path to the original data
        save_data: option to save the new data as .csv, default is False
        path_to_new_data: path to save the new .csv file, required if save_data is True
    Returns: 
        Data frame of cleaned data
        Optionally saves the dataframe as a .csv file according to provided path
         '''
    data = pd.read_csv(path_to_csv)
    dropped_data = data.drop(['Accession num', 'Name'], axis=1)
    #excluding data which is less than 50 amino acids in length
    mask = dropped_data['Sequence'].str.len() > 50  
    only_longer_seqs = dropped_data.loc[mask]
    cleaned_data =  only_longer_seqs[~dropped_data['Sequence'].str.contains('x')]
    sequence_list = []
    for sequence in cleaned_data['Sequence']:
        sequence_list.append(return_encoded_protein_list(sequence))
    cleaned_data['Encoded sequences'] = sequence_list
    cleaned_data['Encoded sequences'] = cleaned_data['Encoded sequences'].apply(np.array)
    print(cleaned_data['Encoded sequences'].dtype)
    if save_data is True:
        if path_to_new_data == False:
            print('path_to_new_data argument must be provided')
        else:
            cleaned_data.to_csv(path_to_new_data)
    return cleaned_data



def pad_list_with_zeros(seq_list, length=None):
    '''Pads list of sequences at end with zeros, according to the longest sequence in the list. 
    Optionally truncates the sequences to a provided length. 
    Args:
        seq_list: the list of sequences to pad
        length (optional): Provides a maximum length to truncate the protein sequences after padding
    Returns:
        List of sequences which have been padded
    '''
    list_with_pads = []
    list_of_seqs = []

    if length is None:
        longest = find_max_list(seq_list)
        for seq in seq_list:
            length_list = len(seq)
            number_pad = longest - length_list
            zero_list = [0] * number_pad
            list_with_pads = seq  + zero_list
            list_of_seqs.append(list_with_pads)
        
    else: 
        longest = length
        for seq in seq_list:
            length_list = len(seq)
            if length_list > length:
                seq = seq[0:length]
            number_pad = longest - length_list
            zero_list = [0] * number_pad
            list_with_pads = seq  + zero_list
            list_of_seqs.append(list_with_pads)
        
        
    return list_of_seqs

def find_max_list(sequence):
    '''Finds the maximum length of a string or list
    Args:
        sequence: list of strings or list of lists
    Returns: Maximum str or list length

    '''
    list_len = [len(i) for i in sequence]
    return max(list_len)
    
def series_string_lists_to_list(data):
    d = pd.read_csv(data)
    seq_list = []

    for seq_list_string in d['Encoded sequences']:
        one_seq_list = literal_eval(seq_list_string)
        seq_list.append(one_seq_list)
    
    return(seq_list)

def return_encoded_protein_list(seq):
    ''' Encodes amino acid sequences into numerical values
    Args:
        seq: Peptide sequeences for conversion to numbers
    Returns: Encoded list
    '''
    char_dict = {'A': 1, 'G': 2, 'I': 3, 'L': 4, 'M': 5, 'W': 6, 'F': 7, 'P': 8, 'V': 9, 'C': 10, 'S': 11, 'T': 12, 'Y': 13, 'N': 14, 'Q': 15, 'H': 16, 'K': 17, 'R': 18, 'D': 19, 'E': 20}
    encoded_list = []
    for index in len(seq):
        character = seq[index]
        encoded_list.append(char_dict[character])
    return encoded_list

class ProteinSeqDataset(Dataset):
    """Creates a data class to allow protein sequence dataset to be iterable for batch feeding the model.
    Returns:
        Tuple of indexable features and labels and the shape shape
        """
    def __init__(self, features_all, label_all):
        assert len(features_all) == len(label_all), "Features and labels must be of equal length."
        #initialise parent class
        super().__init__()  
        self.features_all = features_all 
        self.label_all  = label_all 
    # describes behaviour when the data in the object are indexed
    def __getitem__(self, index):
        #index the features and labels 
        return self.features_all[index], self.label_all[index]        
    # describes behaviour when len is called on the object
    def __len__(self):
        return self.features_all.shape[0]

def split_dataset(  sequences:list, 
                    labels, 
                    random_state:int
                    ):
    """Splits dataset into training, validation and testing datasets
    in the repective ratio of 60:20:20.
    Args:
        dataset (DataFrame): The dataset to be split.
        targets (list): A list of columns to be used as the targets.
        random_state (int): The random state used in the split.
    Returns:
        (tuple): (x_train, y_train, x_validation, y_validation, x_test, y_test)
            in tensor form.
    """
    # dataset_numerical_only = dataset_numerical_only.
    
    x = torch.tensor(sequences)
    print(x.shape)
    y = torch.tensor(labels)
    print(y.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=random_state, shuffle=True)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.5, random_state=random_state)
    return (x_train, y_train, x_validation, y_validation, x_test, y_test)

def create_dataloader(x_train:tensor, y_train:tensor, batch_size:int):
    """Creates a DataLoader from the training dataset.
    Args:
        x_train (tensor): The tensor containing the training features.
        y_train (tensor): The tensor containing the training targets.
        batch_size (int: The size of one batch.
    Returns:
        dataloader (class): DataLoader created from the training set.
    """
    train_dataset = ProteinSeqDataset(x_train, y_train)
    dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class Protein_Seq_NN(torch.nn.Module):
    """A neural network building class.
    Args: 
        kernel: A list of kernel dimensions
    Returns:
        Neural network architecture to be trained
    """
    def __init__(self, kernel):
        super().__init__()
        #define layers
        self.layers = torch.nn.Sequential(
            #changed to 1d as not an image. Num channels, num outputs
            torch.nn.Conv1d(1, 8, kernel[0]),
            torch.nn.ReLU(),
            torch.nn.Conv1d(8, 4, kernel[1]),
            torch.nn.ReLU(),
            
            torch.nn.MaxPool1d(30, 30),
            #This is important.
            torch.nn.Flatten(),
            # LazyLinear removes need for manual calculations of input from the flattened layer
            torch.nn.LazyLinear(5),
            torch.nn.Softmax()
        )
    #define how to stack the different elements of the network together
    def forward(self, X):
            return self.layers(X)

def train_protein_seq_nn(model, num_epochs:int, name_writer:str, dataloader, lr, optimiser):
    """Trains a neural network model
    Args:  
        model: A neural nework model instance
        num_epochs: The number of epochs to train the model on (int)
        name_writer: A name for Tensorboard graph (string)
        dataloader: A dataloader instance for feeding the model
        lr: The learning rate for the optimiser
        optimiser: The optimiser to train the model
    Returns:
        training_metrics: A dictionary containing training_duration and training_loss
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
            #unpack the batch into features and labels
            features, label = batch
            #adding an additional dimension which is required for 1Dconv Due to the required input, this needs to be in the second index
            features = torch.unsqueeze(features, dim=1)
            #make a prediction from the model based on a batch of features
            prediction = model(features)
            #calculate the loss- used to apply to gradient descent algorithm. Using cross entropy 
            training_loss = F.cross_entropy(prediction, label)
            print(f'training_loss: {training_loss}')
            training_acc = multiclass_accuracy(prediction, label)
            training_acc_list.append(training_acc)
            # #populates gradients of model parameters with respect to loss
            diff = training_loss.backward()
            #opimisation step 
            optimiser.step()
            #the .grad associated with tensor .backward does not go back to 0 with every iteration (clearly this would cause issues with SGD), accordingly must re-zero the grad of the optimiser after each iteration. But don't do this before .step!
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
    #calculate the mean entropy loss and accuracy from the last 30 iterations
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
    '''A function to calculate the accuracy of a prediction relative to the label
    Returns: Accuracy as a decimal percentage
    '''
    assert len(labels) == len(prediction)
    n_correct = 0; n_wrong = 0 
    pred_idx = 0; labels_idx = 0
    for i, single_prediction in enumerate(prediction):
        pred_idx = torch.argmax(single_prediction)
        labels_idx = torch.argmax(labels[i]) 
        if pred_idx == labels_idx:
            n_correct += 1
        else:
            n_wrong += 1
    print(f'n_correct: {n_correct}')
    acc = (n_correct * 1.0) / (n_correct + n_wrong)
    return acc

def confusion_matrix_auto(y_true, y_pred, file_name):
    '''Generates a confusion matrix from labels and predictions
    Args:
        y_true: labels
        y_pred: predictions
        file_name: file directory for saving the confusion matrix'''
    # constant for classes
    classes = ('Tyrosine kinase Rec', 'GTP-binding', 'Immunoglobulin heavy', 
            'Histone H', 'Aquaporins')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                        columns = [i for i in classes])
    print(df_cm)
    plt.figure(figsize = (12,7))
    sns.heatmap(df_cm, annot=True)
    plt.show()
    plt.savefig(file_name)

def evaluate_model(model, dataloader_val, path_for_confusion_matrix, dataloader_test=None):
    """Evaluates neural network performance
    Args:
        model: neural network model instance
        dataloader_val: Dataloder instance for the validation dataset
        path_for_confusion_matrix: File directory for saving the generated confusion matrix
        dataloader_test(optional): Dataloder instance for the test dataset
    Returns: Dictionary of performance metrics including: validation loss (mse), validation r2, test loss (mse), test r2, mean inference latency(ms)
        Confusion matrix
        """
    #we don't need the grad function here so switching it off with no_grad
    model.eval()
    model.double()
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
            #adding an additional dimension which is required for 1Dconv, due to the required input, this needs to be in the second index
            x = torch.unsqueeze(x, dim=1)
            #making the prediction
            prediction = model(x)
            #appending the latency into a list
            inference_latencies_list.append(time_ns() - inference_latency_start)
            val_loss = F.cross_entropy(prediction, y)
            print(f'val_loss: {val_loss}')
            #need to instantiate r2 for this to work
            val_acc = multiclass_accuracy(prediction, y)
            #need same dimensions, so flattenned prediction which was (x, 1) dimension
            #prediction_flattened = torch.flatten(prediction)
            val_acc_list.append(val_acc)
            print(f'val_acc: {val_acc}')
            val_loss = val_loss.detach().numpy()
            val_loss_list.append(val_loss)
            #getting index for confusion matrix
            prediction = np.argmax(prediction, axis=1)
            print(prediction)
            y_conf_matrix = []
            for indiv in y: 
                indiv= np.argmax(indiv)
                y_conf_matrix.append(indiv)
            print(y_conf_matrix)
            confusion_matrix_auto(y_conf_matrix, prediction, path_for_confusion_matrix)
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
                # #need same dimensions, so flattenned prediction which was (x, 1) dimension
                # prediction_flattened = torch.flatten(prediction)
                test_acc_list.append(test_acc)
                #truncates the loss list to last 30, with which to get more reliable average
                if len(test_loss_list) > 30:
                    test_loss_list = test_loss_list[-29:]
                test_loss = val_loss.detach().numpy()
                test_loss_list.append(test_loss)
                confusion_matrix(y, prediction, 'Confusion_matrices/best_protein_pred_model_first_pass.png')

    test_loss_mean = np.mean(test_loss_list)
    print(test_loss_mean)
    test_acc_mean = np.mean(test_acc_list)
    print(f'test_acc_mean: {test_acc_mean}')
    mean_inference_latency = np.mean(inference_latencies_list)
    performance_metrics = {'val_loss_mean':val_loss_mean, 'val_acc_mean':val_acc_mean, 'test_loss_mean':test_loss_mean, 'test_acc_mean':test_acc_mean, 'mean_inference_latency':mean_inference_latency}
    return performance_metrics

def get_num_epochs(n_iters:int, batch_size:int, x_train):
    """Calculate number of iterations through the entire training set.
    Args:
        n_iters (int): The number of batches iterated over.
        batch_size (int): The size of one batch.
        x_train (tensor): The tensor containing the training features.
    Returns:
        (int): The number of passes through the entire dataset.
    """
    return int(n_iters / (len(x_train) / batch_size))

def save_nn_model(folder_path:str, model, training_metrics:dict, performance_metrics:dict, all_parameters:dict, hyperparameters=None):
    """Saves neural network model
    Args:
        folder_path: Path to the folder to contain the model
        model: Instance of the neural network to be saved
        performance_metrics: Dictionary of performance metrics
        all_parameters: Dictionary of model parameters
        hyperparametrs: Dictionary of all hyperparameters used to train the model
    Returns: 
        path_date: The time stamp that the model was saved with
        """
    today = datetime.now()
    today = today.strftime('\%Y-%m-%d_%H%M%S')
    print(today)
    os_dir = str(Path(folder_path))
    print(os_dir)
    path_date = str(os_dir + today)
    print(path_date)
    os.mkdir(path_date)
    print(os_dir)
    file_name = str(path_date) + '.pt'
    print(file_name)
    #saving the model
    torch.save(model.state_dict(), file_name)
    #need to convert from tensor to string of json won't save Tensor dtype
    with open(f'{path_date}/parameters.json', 'a') as outfile:
        all_parameters = str(all_parameters)
        json.dump(all_parameters, outfile)
    with open(f'{path_date}/training_metrics.json', 'a') as outfile:
        training_metrics = str(training_metrics)
        json.dump(training_metrics, outfile)
    with open(f'{path_date}/performance_metrics.json', 'a') as outfile:
        performance_metrics =  str(performance_metrics)
        json.dump(performance_metrics, outfile)
    if hyperparameters is not None:
        with open(f'{path_date}/hyperparameters.json', 'a') as outfile:
            hyperparameters = str(hyperparameters)
            json.dump(hyperparameters, outfile)
    return path_date

def generate_nn_config():
    """Creates multiple configuration dictionaries containing hyperparameters for 
    the optimiser, learning rate and depth and width of the hidden
    layers.
    Returns:
        config_dict (dict): Dictionary containing multiple config dictionaries.
    """
    #learning rate values
    learning_rate_tests =   [
                            0.1,
                            0.12,
                            0.08               
                            ]

    protein_length_list =   [100,
                            300, 
                            600
                            ]

    batch_size_list =       [25, 
                            50, 
                            100
                            ]

    kernel_sizes = [[200, 200], [200, 100], [200, 50], [100, 100], [100, 50]]
    
    #declaring the choice of optimisers
    config_dict_list = []
    
    #generating the dictionary
    for learning_rate in learning_rate_tests:
        for protein_length in protein_length_list:
            for batch_size in batch_size_list:
                for kernel in kernel_sizes:
                    one_dict = {}
                    one_dict['learning_rate'] = learning_rate
                    one_dict['protein_length'] = protein_length
                    one_dict['batch_size'] = batch_size
                    one_dict['kernel_sizes'] = kernel
                    config_dict_list.append(one_dict) 
                    
    return config_dict_list

def hyperparameter_screen():
    '''Generates, trains and evaluates different models according to hyperparameters'''
    seq_list = series_string_lists_to_list(r'C:\Users\marko\DS Projects\Machine-learning-with-biological-data\Main_project_file\Prepped data.csv')
    hyperparameters = generate_nn_config()
    print(hyperparameters)
    loss_dict = {}
    for dictionary in hyperparameters:
        print(dictionary)

        padded_seq_list = pad_list_with_zeros(seq_list, dictionary['protein_length'])
        enc = OneHotEncoder() 
        enc.fit(padded_seq_list)
        encoded_padded_list = enc.transform(padded_seq_list).toarray()
        data = pd.read_csv(r'C:\Users\marko\DS Projects\Machine-learning-with-biological-data\Main_project_file\Prepped data.csv')
        enc_2 = OneHotEncoder()
        y = data['Category'].to_numpy()
        y = y.reshape(-1, 1)
        enc_2.fit(y)
        encoded_y = enc_2.transform(y).toarray()
        x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(encoded_padded_list, encoded_y, 42)
        model = Protein_Seq_NN(dictionary['kernel_sizes'])
        n_iters = 1000
        num_epochs = get_num_epochs(n_iters, dictionary['batch_size'], x_train)
        dataloader_train = create_dataloader(x_train, y_train, dictionary['batch_size'])
        training_metrics, model_parameters = train_protein_seq_nn(model, num_epochs, 'protein systematic gridsearch, correct acc 2 lr 1 first', dataloader_train, dictionary['learning_rate'], optimiser=torch.optim.SGD)
        dataloader_val = create_dataloader(x_val, y_val, len(x_val))
        evaluation_metrics = evaluate_model(model, dataloader_val, dataloader_test=None)
        evaluation_metrics['training_loss_mean'] = training_metrics['loss_mean'] 
        evaluation_metrics['training_acc_mean'] = training_metrics['training_acc_mean']
        model_parameters = {"protein_length": dictionary['protein_length'], "batch_size": dictionary['batch_size']} | model_parameters 
        today = datetime.now()
        today = today.strftime('\%Y-%m-%d_%H%M')
        loss_dict[f'{today} training_loss'] = training_metrics["loss_mean"] 
        loss_dict[f'{today} training_loss'] = training_metrics['training_acc_mean']
        loss_dict[f'{today} val_loss_mean'] = evaluation_metrics["val_acc_mean"]
        loss_dict[f'{today} val_acc_mean'] = evaluation_metrics["val_acc_mean"]
        path_date = save_nn_model(r'C:\Users\marko\DS Projects\Machine-learning-with-biological-data\Main_project_file\data and models', model, training_metrics, evaluation_metrics, model_parameters, hyperparameters=dictionary)
    with open(f'{path_date}/aggregated_metrics.json', 'a') as outfile:
        json.dump(loss_dict, outfile)

if __name__ == '__main__':
    pass                 
   
    

    

    