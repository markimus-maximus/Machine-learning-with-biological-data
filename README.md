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

## Predicting cell types from images
