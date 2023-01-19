# Machine-learning-with-biological-data

# Background and aims

This project generated `PyTorch` convoluted neural networks to make predictions about biological data. The first dataset analysed in this project is peptide sequences, with an associated protein category. With these data, a model was trained to predict protein category from the amino acid peptide sequence. The second dataset in this project is a range of microscopic images of different cell types. With this dataset, the model was trained to make predictions of the cell type based on the image presented.

## Making predictions of protein category from amino acid sequences

### The dataset

Data were downloaded from uniprot. The database selected for analysis was the complete human peptide list. From this database, 5 protein categories were chosen comprising of 736 sequences. The 5 categories selected are Tyrosine-protein kinase receptors (n=102), GTP-binding proteins (n=209), immunoglobulin heavy chains (n=134), histone H family proteins (n=216) and aquaporins (n=72).  Once the database was ready, it was necessary to convert the sequences into lists comprising of each element (amino acid residue), and additionally, to improve the potential applicability of the dataset each letter was assigned a numerical integer ranging from 1 to 20 (as below).
`char_dict = {'A': 1, 'G': 2, 'I': 3, 'L': 4, 'M': 5, 'W': 6, 'F': 7, 'P': 8, 'V': 9, 'C': 10, 'S': 11, 'T': 12, 'Y': 13, 'N': 14, 'Q': 15, 'H': 16, 'K': 17, 'R': 18, 'D': 19, 'E': 20}`

The numerical conversion and conversion to list was carried out with the `return_encoded_protein_list(seq)` function which takes a protein sequence (seq) as a string argument. These data were then saved as a .csv file ("Prepped data.csv").

After storing the encoded data, the next task was to generate a list of list sequences from the .csv file. To achieve this, the function `series_string_lists_to_list(data)` was created. A consequence of storing lists as a .csv file is that they appear as strings, so the `literal_eval` method was used to convert back to lists, before being concatenated into a list of lists. 

Neural networks generally require consistent input size, and accordingly the length of the protein sequences had to be standardised. This was achieved by padding the end of sequences lower than the maximimum with zeros, a common padding approach. The `pad_list_with_zeros(seq_list, length=None)` function was created to achieve the zero padding. The arguments required are a sequence list, and an optional parameter for the length of the protein. If the length argument is not defined when the function is called, the function will pad all sequences up to the maximum length within the dataset. If the length is declared, all sequences are truncated to that length after padding has occurred. 

## Predicting cell types from images
