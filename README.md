# Machine-learning-with-biological-data

# Background and aims

This project generated `PyTorch` convoluted neural networks to make predictions about biological data. The first dataset analysed in this project is peptide sequences, with an associated protein category. With these data, a model was trained to predict protein category from the amino acid peptide sequence. The second dataset in this project is a range of microscopic images of different cell types. With this dataset, the model was trained to make predictions of the cell type based on the image presented.

## Making predictions of protein category from amino acid sequences

### Obtaining and preparing the dataset

Data were downloaded from uniprot. The database selected for analysis was the complete human peptide list. From this database, 5 protein categories were chosen comprising of 736 sequences. The 5 categories selected are Tyrosine-protein kinase receptors (n=102), GTP-binding proteins (n=209), immunoglobulin heavy chains (n=134), histone H family proteins (n=216) and aquaporins (n=72).  Once the database was ready, it was necessary to convert the sequences into lists comprising of each element (amino acid residue), and additionally, to improve the potential applicability of the dataset each letter was assigned a numerical integer ranging from 1 to 20 (as below).
`char_dict = {'A': 1, 'G': 2, 'I': 3, 'L': 4, 'M': 5, 'W': 6, 'F': 7, 'P': 8, 'V': 9, 'C': 10, 'S': 11, 'T': 12, 'Y': 13, 'N': 14, 'Q': 15, 'H': 16, 'K': 17, 'R': 18, 'D': 19, 'E': 20}`

The numerical conversion and conversion to list was carried out with the `return_encoded_protein_list(seq)` function which takes a protein sequence (seq) as a string argument. These data were then saved as a .csv file ("Prepped data.csv").

After storing the encoded data, the next task was to generate a list of list sequences from the .csv file. To achieve this, the function `series_string_lists_to_list(data)` was created. A consequence of storing lists as a .csv file is that they appear as strings, so the `literal_eval` method was used to convert back to lists, before being concatenated into a list of lists. 

Neural networks generally require consistent input size, and accordingly the length of the protein sequences had to be standardised. This was achieved by padding the end of sequences lower than the maximimum with zeros, a common padding approach. The `pad_list_with_zeros(seq_list, length=None)` function was created to achieve the zero padding. The arguments required are a sequence list, and an optional parameter for the length of the protein. If the length argument is not defined when the function is called, the function will pad all sequences up to the maximum length within the dataset. If the length is declared, all sequences are truncated to that length after padding has occurred. Returned is a padded dataset in list form.

Once a list of padded sequences have been generated, the data were one hot encoded. This generates data in a form required by PyTorch, while also decreasing the possibility of weight biases if other numerical data were used to train the model. The labels were also one hot encoded.

Splitting data into discrete groups ahead of model training ensures that the model has not seen the data when it is later applied for validation and testing purposes. To split the data, the `split_dataset(sequences:list, labels, random_state:int)` function was written. In addition to splitting the data, this function also converts the datasets into `Tensor`, the data type required to train `PyTorch` models.
 
 The final piece of data preparation was to generate a dataset class which allows the datset to be indexed and tprovdes a length, this making the data iterable. This iterability allows the data to be fed in batches. See below for the dataset class.
 ~~~
 class ProteinSeqDataset(Dataset):
    """Creates a data class to allow a given dataset to be iterable for batch feeding the model.
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
 ~~~

### Generating and training the model

The `create_dataloader` function was created to batch feed the model data. To achieve this the class `ProteinSeqDataset` described above was used, as well as the `Dataloader` class from PyTorch. 

Given that the sequences are 1 dimensional, 1d convolutional layers were selected for the nerual network. The preliminary architecture of the model for screening purposes comprised of 2x 1dconvoluted layes, 1x maxpool layer (max 30), and a fully connected linear regression layer. The code including activator functions is shown below.

~~~
def __init__(self):
        super().__init__()
        #define layers
        self.layers = torch.nn.Sequential(
            #changed to 1d as not an image. Num channels, num outputs
            torch.nn.Conv1d(1, 4, 200),
            torch.nn.ReLU(),
            torch.nn.Conv1d(4, 8, 50),
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
~~~

To iteratively train the model, the `train_protein_seq_nn(model, num_epochs:int, name_writer:str, dataloader, lr, optimiser)` function was coded. `torch.unsqueeze` was used to get the feature `Tensor` in the right shape for calculating loss. This function uses `cross_entropy` as the criterion to iteratively optimise the model. The loss of the final 30 iterations of the model are taken as an average to calculate the training loss, reflecting loss at the most optimised point in the training process. `Tensorboard` is used in this function to monitor the training process in real time. This function returns a tuple containing `training_metrics` which is a dictionary containing the training duration and the average loss, and `model_parameters` which is a dictionary which contains the model and optimiser state parameters.

### Evaluating and optimising the model

In order to assess applicability of the model, unseen datasets were used to evaluate the model (generated with a separate dataloader instance). As with the training metrics, the criterion for loss for evaluating the model was generated with `cross_entropy`. A helper function `multiclass_accuracy` was generated to calculate the accuracy of the model in making predictions from the validation and testing datasets. The evaluation function provides optional additional functionality should the user want to include a testing dataset also. The function returns a dictionary of performance metrics which contain the mean loss and accuracy for the validation dataset and, optionally, the testing dataset. The evaluation metrics also contain the mean inference latency (time taken to evaluate the loss and accuracy).

In the first instance, the model was optimised by varying hyperparameters to identify those which return the best evaluation metrics. The `generate_nn_config` function was coded to return a list of dictionaries with which to iterate through, with each dictionary containing a unique combination of the hyperparameters. The hyperparameters assessed were the learning rate (0.3, 0.1, 0.05), the protein length fed into the model (100, 200, 400, 600 amino acid length) and the batch size (25, 60, 100). The function `hyperparameter_screen` was coded to bring together all of the above functionality and allow for iterative execution of the functionality with respect to each unique hyperparameter combination. After each iteration, the training metric dictionary, model and optimisation parameters dicitionary, and evaluation metrics dictionary were saved as json files and named to include the date and time, using the `save_nn_model` function. Simimlarly the model data were saved with this function. Finally, an aggregated dictionary of all of the evaluation metrics was generated in order to easily extract and compare the data after all of the vombinations of hyperparameters had been carried out. 



## Predicting cell types from images
