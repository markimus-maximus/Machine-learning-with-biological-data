# Machine-learning-with-biological-data

# Background and aims

This project generated `PyTorch` convoluted neural networks to make predictions about biological data. The first dataset analysed in this project is peptide sequences, with an associated protein category. With these data, a model was trained to predict protein category from the amino acid peptide sequence. The second dataset in this project is a range of microscopic images of different cell types. With this dataset, the model was trained to make predictions of the cell type based on the image presented. The third dataset is a series of blood sample images containing 1 of 4 variants of white blood cells (eosinophils, lymphocytes, monocytes and neutrophils), with the aim to correctly identify the variant from the images.

## Making predictions of protein category from amino acid sequences

### Obtaining and preparing the dataset

All code for this part of the project can be found in `Protein_sequences.py`. The database selected for analysis was the complete human peptide list from uniprot. To initially process the data, `prep_protein_seq_data(path_to_csv, save_data=False, path_to_new_data=False)` was generated. Using this function, all sequences shorter than 50 amino acid residues were excluded and similarly, those sequences containing 'X', where the amino acid sequences was unknown, were also excluded.  From this database, 5 protein categories were chosen comprising of 736 sequences. The 5 categories selected are Tyrosine-protein kinase receptors (n=102), GTP-binding proteins (n=209), immunoglobulin heavy chains (n=134), histone H family proteins (n=216) and aquaporins (n=72). See image below for screenshot of the dataset

![image](https://user-images.githubusercontent.com/107410852/213874794-da1fa551-65b6-4003-b3fe-4a76f8120c3a.png)

Once the database was ready, it was necessary to convert the sequences into lists comprising of each element (amino acid residue), and additionally, to improve the potential applicability of the dataset each letter was assigned a numerical integer ranging from 1 to 20 (as below). This was achieved by generating the `return_encoded_protein_list(seq)` function. 
`char_dict = {'A': 1, 'G': 2, 'I': 3, 'L': 4, 'M': 5, 'W': 6, 'F': 7, 'P': 8, 'V': 9, 'C': 10, 'S': 11, 'T': 12, 'Y': 13, 'N': 14, 'Q': 15, 'H': 16, 'K': 17, 'R': 18, 'D': 19, 'E': 20}`

The numerical conversion and conversion to list was carried out with the `return_encoded_protein_list(seq)` function which takes a protein sequence (seq) as a string argument. These data were then saved as a .csv file ("Prepped data.csv").

After storing the encoded data, the next task was to generate a list of list sequences from the .csv file. To achieve this, the function `series_string_lists_to_list(data)` was created. A consequence of storing lists as a .csv file is that they appear as strings, so the `literal_eval` method was used to convert back to lists, before being concatenated into a list of lists. 

Neural networks generally require consistent input size, and accordingly the length of the protein sequences had to be standardised. This was achieved by padding the end of sequences lower than the maximimum with zeros, a common padding approach. The `pad_list_with_zeros(seq_list, length=None)` function was created to achieve the zero padding. The arguments required are a sequence list, and an optional parameter for the length of the protein. If the length argument is not defined when the function is called, the function will pad all sequences up to the maximum length within the dataset. If the length is declared, all sequences are truncated to that length after padding has occurred. Returned is a padded dataset in list form.

Once a list of padded sequences had been generated, the data were one hot encoded. This encoding generates data in a form required by PyTorch, while also decreasing the possibility of weight biases if other numerical data were used to train the model. The labels were also one hot encoded.

Splitting data into discrete groups ahead of model training ensures that the model has not seen the data when it is later applied for validation and testing purposes. To split the data, the `split_dataset(sequences:list, labels, random_state:int)` function was written. In addition to splitting the data, this function also converts the datasets into `Tensor`, the data type required to train `PyTorch` models.

Subsequent code can be found in Cell_image_PyTorch.py
 
The final piece of data preparation was to generate a dataset class which allows the datset to be indexed and provides a length, thus making the data iterable. This iterability allows the data to be fed in batches. As always, dataset classes inherit from torch.utils.data.Dataset. See below for the dataset class.
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

Given that the sequences are 1 dimensional, 1d convolutional layers were selected for the nerual network. The preliminary architecture of the model for screening purposes comprised of 2x 1dconvoluted layes, 1x maxpool layer (max 30), and a fully connected linear regression layer. The architecture of the nerual network including activator functions is shown below.

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

In addition to generating evaluation metrics, a confusion matrix is also generated with each evaluation. This ws achieved by writing `confusion_matrix_auto(y_true, y_pred, file_name)`. The `np.arg_max` was built into the evaluation method to return the index of both the labels and predictions, with which to update the confusion matrix. This function uses the `sk-learn` `confusion_matrix` method to compute the confusion matrix. A directory is provided to save the confusion matrix. 

In the first instance, the model was optimised by varying hyperparameters to identify those which return the best evaluation metrics. A quick broad scan of learning rate showed that ~0.1 was the best for this approach, and this was used as a preliminary learning rate for hyperparameter screening (to reduce the number of combinations of hyperparameters). The `generate_nn_config` function was coded to return a list of dictionaries with which to iterate through, with each dictionary containing a unique combination of the hyperparameters. The hyperparameters assessed were the the protein length fed into the model (100 and 300 amino acid length) the batch size (25, 60, 100), and the kernel sizes ([200, 200], [200, 100], [200, 50], [100, 100], [100, 50]). The function `hyperparameter_screen` was coded to bring together all of the above functionality and allow for iterative execution of the functionality with respect to each unique hyperparameter combination. After each iteration, the training metric dictionary, model and optimisation parameters dicitionary, and evaluation metrics dictionary were saved as json files and named to include the date and time, using the `save_nn_model` function. Simimlarly the model data were saved with this function. Finally, an aggregated dictionary of all of the evaluation metrics was generated in order to easily extract and compare the data after all of the vombinations of hyperparameters had been carried out. 

After screening the different hyperparameters, the data for the hyperparameters with the top 10 cross entropy scores were displayed (below).
![image](https://user-images.githubusercontent.com/107410852/213686575-18733a49-6b19-454d-9450-6ac18fb167d6.png)

A graph of all of the learning curves for the optimisation step is shown below:
![image](https://user-images.githubusercontent.com/107410852/213875710-8c2b99df-385d-4d3f-aa7a-465b6dcbbe45.png)

In general, the best performing models were generating a ~65 % accuracy on the validation dataset, which is promising for a first pass attempt. For all models there was evidently an element of overfitting, as the accuracy for the training set was around maximum. Some regularisation approaches could be useful for future additions.

In terms of optimal hyperparameters, for protein length and batch size there was not a clear trend. The kernel sizes were, however, more likely to impact the high performers, with 5/10 of the top-performing models having kernel sizes [100, 100]. 

![image](https://user-images.githubusercontent.com/107410852/213687931-a4ab465f-0fbc-4a57-9d34-6d9ba9f4fa94.png)

The kernel values of [100,100] will be used as a basis for future tests. Future optimisations will seek to understand the effect of variations to learning rate and neural network architecture.

After reloading the best performing model, a confusion matrix was generated from the validation dataset, which showed that for each category, the correct choice was selected the most. The model performed particularly well in predicting immunoglobulins, probably because there are highly conserved canonical sequences in this family. The tyrosine kinase receptor and Histone H sequences were also well-predicted. A poorly predicted category was the aquaporins, however, this was the smallest set of data used to train the model, which may have contributed to this.  Nonetheless, this category were still predicted correctly 50 % of the time, which is not bad for a first pass. GTP-binding proteins were the poorest predicted of all the categories, pointing to non-canonical sequence structures. 

![protein_pred_first_pass](https://user-images.githubusercontent.com/107410852/213874133-0f2332e1-3103-4e5f-939b-8dbd4e55e6ac.png)

### Future optimisations and concluding remarks

Considering the model architecture was quite narrow with two convolutional layers, this could be an area to improve for future optimisation rounds. Indeed, other approaches in the literature for making predictions from protein sequences used 10 convoluted networks. Fine-tuning of the learning rate could also be an avenue for improvement. Due to the GPU of the laptop used for the modelling, the sequence length possible to plug into the model was quite limited to 600 residues without crashing the machine. An ability to increase the protein length to greater than 600 residues could well improve the predictive power of models, particularly for proteins with canonical sequences towards the end of the proteins. It could also be that slicicing the protein sequences in a region which was not at the start of the protein (currently proteins are trunated from the end) may yield more interesting information for certain protein families. This protein slicing could be informed in accordance to approximate locations of highly conserved regions (for example GPCRs are likely to have conserved regions towards the carboxyl end). There was a clear amount of overfitting occurring, so early stopping of the training, informed by learning curves, could be an approach to reduce this. Other approaches to prevent overfitting could include variations to dropout.  Despite the areas for improvement, for a first pass gaining an accuracy of ~65 % for this problem seems encouraging.

## Predicting cell types from images

This section of work aimed to generate 2d convoluted neural networks to make predictions of cell type based on images. A custom dataset of red blood cells and neurons was generated, containing 50 images each. Example images of each category are shown below:

![image](https://user-images.githubusercontent.com/107410852/213875863-952a910d-4890-41f2-bf18-842cb5def346.png)

Red blood cells

![image](https://user-images.githubusercontent.com/107410852/213875896-6fb73d3e-5701-48db-9509-ed0e35e2e11e.png)

Neurons

### Generating utility functions to prepare images for model

The utility functions for image processing can be found in `prepare_image_data.py`. The file names and associated categories were first added to a .csv file. This file was then fed into the  `split_data_into_dataset(path_to_csv, train_to_dev_ratio,  val_to_test_ratio)` function which takes these data and, using `pd.dataframe.sample`, randomly shuffles the data. The data are then split into the train, val and test datasets according to the `train_to_dev_ratio` and `val_to_test_ratio` arguments. The function then generates .csv files for each of the datasets. 

Once the names and categories of the images had been split, the images themselves required splitting in the same manner. To achieve this, `partition_images(source_folder, destination_folder, data_file_directory)`was written. A .csv file corresponding to one of the datasets (generated above) is used as a reference image name list. The images are only copied to the destination folder if the name of the image is found in the reference dataset. 

### Creating an iterable dataset for feeding the model

In order to train the model, the dataset prepared above needs to be made into a class which returns its len and allows for indexing. As always, dataset classes inherit from torch.utils.data.Dataset. Generating this dataset was more involved for images than for tabular data (generated above), due to the factthat the features were images. Accordingly, `cv2.imread` was utilised to import the image data, and this was converted to a tensor.  See the code below.

~~~
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
~~~
Due to the image data being multiple files instead of one file as with tabular data, the containing folder of the images is required, as well as the name of the file (in the respective .csv file). The image must also be read and converted to an array, then to a tensor. There is an option to transform the images. With these data, the images were resized, converted to grayscale and normalised using `transforms.Compose` as below. The reason for grayscaling the images was that, depending on the type of staining and/or fluorophores used, images of the same cell type can be any colour, and as such colour was anticipated to be unhelpful and potentially misleading. Normalisation was applied to compensate for variations in photo brightness which could strongly bias certain phots above others. Thus, normalising allows for better model characterisation of shapes within the images.
~~~
transforms = T.Compose([
                        T.Resize([62, 156]),
                        T.Grayscale(),
                        T.ToTensor(),
                        T.Normalize((0.5, ), (0.5, ))
                        ])
~~~
To get the lowest dimensions of the dataset with which to carry out the resizing, `get_lowest_image_dimensions_from_folder(original_image_folder_directory:str)` utility funtion was generated and utilised.

### Generating, training and evaluating the model
The architecture of the model is largely the same as the protein prediction architecture, except that 2d convolutional layers and maxpool were 2d instead of 1d. Equally, the code for the training loop and the evaluation was very similar to the protein prediction. 

Training curves (below) show that there was a very good training loss generated from the training. Indeed, variations to the kernel size and learning rate resulted in 95 % accuracy from the validation set. The optimal conditions were a learning rate of 0.1 and kernel sizes of 8 for both convolutional layers (the green line below). 

![image](https://user-images.githubusercontent.com/107410852/213882842-0c9735ad-f9c4-4474-81ca-fac1aec45bab.png)

Predicting cell type from images was successful with little need for optimisations. An interesting follow up to this project would be to add yet more cell types into the dataset to really test the predictive power of the neural networks.

## Classifying white blood cells

A Kaggle dataset was used for this element of work (https://www.kaggle.com/datasets/paultimothymooney/blood-cells). There are images of 4 types of blood images containing white blood cells (eosinophils, lymphocyts, monocytes, neutrophils). Each class of white blood cells was contained in an individual folder, meaning that `torchvision.datasets.ImageFolder` could be utilised to generate an iterable dataset. Below is the code for generating the dataset, including the normalisations. Normalisations were applied according to the mean intensities and intensity standard deviations of the dataset, to allow for more efficient training of the model. 

~~~
train_dataset = ImageFolder(
    root=r'C:\Users\marko\DS Projects\Machine-learning-with-biological-data\Main_project_file\Images\White blood cells kaggle dataset\dataset2-master\archive\dataset2-master\dataset2-master\images\TRAIN',
    transform=T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
~~~

`DataLoader` was generated from `torch.utils.data`. A new configurable `WBCs_CNN` convoluted neural network architecture was created, as below: 

~~~
def __init__(self, kernel, input_dim, output_dim):
        super().__init__()
            #define layers
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim, 8, kernel[0]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, kernel[1]),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(30, 30),
            #This is important.
            torch.nn.Flatten(),
            # LazyLinear removes need for manual calculations of input from the flattened layer
            torch.nn.LazyLinear(output_dim),
            torch.nn.Softmax()
        )
        #define how to stack the different elements of the network together
    def forward(self, X):
            return self.layers(X)
~~~

The functions for the training and evaluation loops are similar to the previous examples shown here. For the evaluation loop, there was additional functionality added to batch feed with more than 1 iteration due to the size of the dataset and the associated computational demand. The optional argument to include a test dataset was also removed from the evaluation function since there is no test dataset (only validation). The saving function uses similar functionality as previously, except the model parameters were excluded from saving due to inability to serialise the data into a json file. This exclusion of the model parameters should not be an issue since the model parameters should be saved when the model is saved. To screen hyperparameters, a variation of the previous functions was generated, see below for the code: 

~~~
def hyperparameter_screen(train_dataset, val_dataset):
    '''Generates, trains and evaluates different models according to hyperparameters'''
    hyperparameters = generate_nn_config()
    
    for dictionary in hyperparameters:
        print(f'next hyperparameters: {dictionary}')
        batch_size = dictionary['batch_size'] 
        
        dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        #controls the device allocation, GPU preferred
        NN_inst = WBCs_CNN([7, 7], input_dim=3, output_dim=4).to(device)
        
        training_metrics = train_WBCs_nn(NN_inst, 10, 'WBC detection screen', dataloader_train, dictionary['learning_rate'], torch.optim.SGD)
        eval_batch_size = 200
        evaluation_metrics = evaluate_WBCs_model(NN_inst, dataloader_val, eval_batch_size, val_dataset)
        save_model_and_data(r'C:\Users\marko\DS Projects\Machine-learning-with-biological-data\Main_project_file\data_models_WBCs', NN_inst, training_metrics, evaluation_metrics, dictionary)
~~~
The first 2 hyperparameters which were screened were learning rate (0.08, 1, 1.2) and batch size (50, 100, 150). Across all the hyperparameter combinations, there was a range in accuracy from 29.1 % to 55.1 % (see below for all hyperparamter combinations). The best combination of hyperparameters observed with the validation dataset was learning rate of 0.08 and batch size of 100. The second and third best hyperparameter combinations were 0.08 learning rate with batch size 50 and learning rate of 1 with batch size 100, respectively. These findings show that learning rate of 0.08  is the best of those tested for training these data.

![image](https://user-images.githubusercontent.com/107410852/215767168-2d3a422d-ee0d-4624-9134-db5fcd4b35b5.png)

Interestingly, the best performing model in terms of training loss was learning rate of 1 with batch size of 50. However, this model showed a larger amount of overfitting. In contrast, while having poorer training losses, the best performing models with learning rate of 0.08 generalised much better. See below for the generalisation scores, calculated as training loss divided by validation loss.

![image](https://user-images.githubusercontent.com/107410852/215768395-529b2adf-9a94-4925-bd5f-50c404ddb7cc.png)

The best-performing model was able to predict category of white blood cells far better than random chance (55.1 and 25 %, respectively). However, there is still room for improvement, and as such there are still other approaches which could be employed. One such approach would be to further optimise the batch size and learning rate. Another would be to explore decreasing overfitting by examining the effect of number of epochs used to train, or utilising dropout approaches. Changes to the kernel size and/or adding further neural network layers could also be implemented to improve the predictive power. 






