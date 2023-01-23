from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import glob
import os
import pandas as pd
import shutil
import torchvision.transforms as transforms

def get_lowest_image_dimensions_from_folder(original_image_folder_directory:str):
    '''Finds the lowest height and width dimensions from a set of images in a directory.
    Args: 
        original_image_folder_directory: folder containing the images to be sized
    Returns:
        printed lowest height and width dimensions '''

    list_of_all_images_height = []
    list_of_all_images_width = []
    for subdir, dirs, files in os.walk(original_image_folder_directory):
        for file in files:
                print(f'Reading file {file}')
                image = Image.open(os.path.join(subdir, file))
                image_width = int(image.size[0])
                image_height = int(image.size[1])
                aspect_ratio = image_height / image_width
                list_of_all_images_height.append(image_height)
                list_of_all_images_width.append(image_width)
    min_height = min(list_of_all_images_height)
    min_width = min(list_of_all_images_width)
    print (f'Lowest image width is {min_width} and lowest image height is {min_height}')

def partition_images(source_folder, destination_folder, data_file_directory):    
    '''Patitions images into separated folders according to the dataset provided
    Args:
        source_folder: the folder containing the images for partitioning
        destination_folder: the folder for partitioned files
        data_file_directory: The file containing the relevant dataset
    Returns:
        A new folder containing images according to a dataset'''       
    files = os.listdir(source_folder)
    df = pd.read_csv(data_file_directory)
    dataset_names = list(df.iloc[:,0])
    for file in files:
        full_name = os.path.join(source_folder, file)
        if file in dataset_names:
            shutil.copy(full_name, destination_folder)
        
def split_data_into_datasets(path_to_csv, 
                            train_to_dev_ratio, 
                            val_to_test_ratio
                            ):
    ''' Generates 3 dataframes and .csv files of train, validation and test datasets
    Args:
        path_to_csv: The .csv file containing data to be split
        train_to_dev_ratio: the ratio of training to devlopment (val and test) datasets
        val_to_test_ratio: the ratio of val to test datasets
    Returns:
        3 saved .csv files corresponding to the split datasets'''
    df = pd.read_csv(path_to_csv)
    df = df.sample(frac=1).reset_index(drop=True)
    df_length = len(df)
    len_train = int(df_length*train_to_dev_ratio)
    train_df = df[:len_train].reset_index(drop=True)
    dev_df = df[len_train:]
    len_dev = len(dev_df)
    len_val = int(len_dev * val_to_test_ratio)
    val_df = dev_df[:len_val].reset_index(drop=True)
    test_df = dev_df[len_val:].reset_index(drop=True)
    train_df.to_csv('train_dataset.csv', index=False, header=False)
    val_df.to_csv('val_dataset.csv', index=False, header=False)
    test_df.to_csv('test_dataset.csv', index=False, header=False)  


def get_num_epochs(n_iters:int, batch_size:int, x_train):
    """Calculate number of passes through the entire training set.
    Args:
        n_iters (int): The number of batches iterated over.
        batch_size (int): The size of one batch.
        x_train (tensor): The tensor containing the training features.
    Returns:
        (int): The number of passes through the entire dataset.
    """
    return int(n_iters / (len(x_train) / batch_size))

if __name__ == '__main__': 
    pass  
