import os
import gdown
import glob
import math
import torch
import shutil
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda

import numpy as np

import pandas as pd
from torchvision.io import read_image
from zipfile import ZipFile

import matplotlib.pyplot as plt

class DvmCarDataset(Dataset):
    """PyTorch wrapper for DVM-CAR dataset.

    Args:
        Dataset (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Default local work directory
    work_def = 'dvmcar'    
    persist_def = None

    # Default download url
    url_def = 'https://figshare.com/ndownloader/articles/19586296/versions/1'

    def __init__(self,
                work: str = work_def,
                persist: str = persist_def,
                url: str = url_def,
                split = [0, 1],
                transform = None):

        """Construct a DvmCarDataset using specified parameters. If dvmcar.zip 
        does not exist at the local work location, the constructor attempts to
        copy it from persist and if that is unsuccessful downloads it from url
        to both work and persist. The constructor then unpacks the three
        subordinate zip files from dvmcar.zip and expands each of them. Two of
        the zip files contain images and the third contains supporting csv
        files. The constructor forms a pandas dataframe with the full paths to
        every image and various other fields extracted from the image filename.
        It joins this with the image table in order to provide means of
        navigating from image id to path (though in practice we may not require
        that linkage). As the original class labels ("Genmodel ID") are strings,
        the constructor forms integer ids based on alphabetical ordering of
        unique labels and adds a column with these integer labels to the table.

        The getitem method optionally applies a transform to the image.

        Args:
            split (list, optional): specifies the dataset split to be returned
            as a list with lower and upper limits
            transform (_type_, optional): specifies getitem transform. Defaults
            to None. 
            work (str, optional): specifies path to dvmcar.zip. Defaults to work_def. 
            persist (str, optional): specifies path to persistent storage of
            dvmcar.zip. Defaults to None. 
            url (str, optional): specifies downlaod url for dvmcar.zip. Defaults
            to url_def.
        """

        # Retain arguments
        self.split = split
        self.work = work
        self.persist = persist
        self.url = url
        self.transform = transform

        # Fetch data to work file
        DvmCarDataset.fetch_zip(work, persist, url)

        # Unpack work file
        DvmCarDataset.unpack_zip(work)

        # Store work directory
        self.work_dir = os.path.split(work)[0]

        # Load useful data tables as pandas dataframes
        # At present we only require Image_table.

        # self.ad_df = pd.read_csv(os.path.join(
        #     self.work_dir, 'Ad_table.csv'), skipinitialspace=True)
        self.basic_df = pd.read_csv(os.path.join(
            self.work_dir, 'Basic_table.csv'), skipinitialspace=True)
        self.image_df = pd.read_csv(os.path.join(
            self.work_dir, 'Image_table.csv'), skipinitialspace=True)
        # self.price_df = pd.read_csv(os.path.join(
        #     self.work_dir, 'Price_table.csv'), skipinitialspace=True)
        # self.sales_df = pd.read_csv(os.path.join(
        #     self.work_dir, 'Sales_table.csv'), skipinitialspace=True)
        # self.trim_df = pd.read_csv(os.path.join(
        #     self.work_dir, 'Trim_table.csv'), skipinitialspace=True)

        # Get classes from basic table
        self.classes = len(self.basic_df)
        self.class_id = list(self.basic_df['Genmodel_ID'])
        self.class_name = list(self.basic_df['Genmodel'])
        self.class_id_to_index = dict(
            zip(list(self.class_id), range(len(self.basic_df))))

        # Get image info as a pandas dataframe
        self.info_df = DvmCarDataset.load_image_info(
            work, ['resized_DVM', 'confirmed_fronts'], self.class_id_to_index)

        # Drop duplicate images
        self.info_df.drop_duplicates(subset=['Image_name'], inplace=True)

        # # Join info with image table for viewpoint, quality
        self.info_df = pd.merge(
            self.info_df, self.image_df, left_on='Image_name', right_on='Image_name')

        # Sort on image name to be sure of order
        self.info_df.sort_values(by='Image_name', inplace=True)

        # Create a random number generator with repeatable seed
        rng = np.random.default_rng(seed=0)

        # Create shuffled order for data splits
        self.shuffle = rng.permutation(len(self.info_df))

        # Set the dataset split
        self.set_split(split)

    def __len__(self):
        """Returns number of images in the dataset split.

        Returns:
            _type_: Image count.
        """

        # Number of rows in the dataframe split
        return self.i1-self.i0

    def __getitem__(self, index: int):
        """Retrieve the indexed record from the dataset.

        Args:
            idx (int): specifies the item index

        Returns:
            (image, label): returns image and corresponding label 
        """

        # Randomly permute the index
        shuffled_index = self.shuffle[index+self.i0]

        # Retrieve path to image
        img_path = self.info_df['Image_path'][shuffled_index]

        # Read the image
        image = read_image(img_path).float()

        # Get the label number
        label = self.info_df['class_index'][shuffled_index]

        # If transform specified...
        if self.transform:

            # Apply transform
            image = self.transform(image)

        # Return image and label
        return image, label

    def set_split(self, split):

        # Save split 
        self.split = split

        # Set split limits
        self.i0 = math.floor(split[0]*len(self.info_df))
        self.i1 = math.floor(split[1]*len(self.info_df))        

    @classmethod
    def fetch_zip(cls, work: str, persist: str, url: str):
        """If the zip file specified by work does not exist, attempt to
        copy it from persist. If that doesn't work attempt to download it from
        url to work and optionally copy it to persist. Return true if the file
        exists at work on exit.

        Args:
            work (str): specifies full path to dvmcar.zip
            persist (str): optionally specifies full path to persistent copy of
            dvmcar.zip
            url (str): specifies url from which to download dvmcar.zip in liue
            of work and persist 

        Returns:
            Boolean: True if work has dvmcar.zip on exit
        """

        try:

            # Split work path
            work_dir = os.path.split(work)[0]

            # Coerce work directory into existence
            os.makedirs(work_dir, exist_ok=True)

            # If work file already exists...
            if os.path.exists(work):

                # Log progress
                print('Work file {0} is already available.'.format(work))

                # Nothing to do
                return True

            # If persist file exists...
            if persist is not None and os.path.exists(persist):

                # Copy from persist to work
                shutil.copyfile(persist, work)
                      
                # Log progress
                print('Copied work file from {0} to {1}.'.format(
                    persist, work))

                # Nothing more to do
                return True

            # Download from url to work
            gdown.download(url, work, quiet=False)

            # Log progress
            print('Downloaded work file from {0} to {1}.'.format(url, work))

            # If persist specified...
            if persist is not None:

                # Split persist path
                persist_dir = os.path.split(persist)[0]

                # Coerce work directory into existence
                os.makedirs(persist_dir, exist_ok=True)

                # Copy from work to persist
                shutil.copyfile(work, persist)

                # Log progress
                print('Copied work file from {0} to {1}.'.format(
                    work, persist))

                # Nothing more to do
                return True

        except:

            # Log progress
            print('Failed to fetch {0}.'.format(work))

            # No joy
            return False

    @classmethod
    def unpack_zip(cls, work: str):
        """Unpack dvmcar.zip into three zipfiles and then unpack each of these.

        Args:
            work (str): specifies location of dvmcar.zip
        """

        # Split work file path
        work_dir = os.path.split(work)[0]

        # Open the work zip file
        with ZipFile(work, 'r') as work_zip:

            # Get inner file names
            inner_names = work_zip.namelist()

            # For each inner file name...
            for inner_name in inner_names:

                # Create full file name
                full = os.path.join(work_dir, inner_name)

                # If zip file doesn't exist...
                if os.path.exists(full):

                    # Log use of existing file
                    print('Using existing {0}.'.format(full))

                else:

                    # Log use of existing file
                    print('Extracting {0}.'.format(full))

                    # Extract it
                    work_zip.extract(inner_name, work_dir)

                    # Log use of existing file
                    print('Unpacking contents of {0}.'.format(full))

                    # Extract its contents
                    with ZipFile(full) as inner_zip:

                        # Extract its contents
                        inner_zip.extractall(work_dir)

    @classmethod
    def load_image_info(cls, work: str, dirs: list, class_id_to_index: dict):
        """If info.csv exists in same directory as dvmcar.zip specified by work,
        load it. If not, iterate over directories specified in dirs finding all
        jpg files recursively. Extract fields from the name of each jpg file.

        Args:
            work (str): specifies the path to dvmcar.zip
            dirs (list): specifies a list of directories containing jpg images
        """

        # Get work directory
        work_dir = os.path.split(work)[0]

        # Form csv file path
        csv = os.path.join(work_dir, 'info.csv')

        # If csv file exists...
        if os.path.exists(csv):

            # Read info from csv
            info = pd.read_csv(csv)

        else:

            # Create lists for jpg paths
            jpg_paths = []

            # For each directory...
            for d in dirs:

                # Form path to direcrtory
                dir_path = os.path.join(work_dir, d)

                # Find all files ending in .jpg recursively and add them to the jpg path list
                jpg_paths += glob.glob(dir_path + '/**/*.jpg', recursive=True)

            # Example filename
            # Nissan$$Pulsar$$2016$$Grey$$64_32$$457$$image_10.jpg

            # Create list to hold image file records
            records = []

            # Create a set to hold unique string labels
            genmodel_ids = set()

            # For each jpg path in the list of jpg paths...
            for jpg_path in jpg_paths:

                # Extract the name of the jpg file
                jpg_name = os.path.split(jpg_path)[1]

                # Split the name into fields
                fields = jpg_name.split('$$')

                # Extract fields from the image name
                maker = fields[0]
                genmodel = fields[1]
                year = fields[2]
                color = fields[3]
                genmodel_id = fields[4]
                instance = fields[5]
                imname = fields[6]

                # Look up class index
                class_index = class_id_to_index[genmodel_id]

                # Append a row with fields
                records.append([jpg_path, jpg_name, maker, genmodel,
                                year, color, genmodel_id, instance, class_index])

                # Add the "Genmodel_ID" field to to the list of class labels
                genmodel_ids.add(fields[4])

            # Create dataframe
            info = pd.DataFrame(records, columns=[
                                'Image_path', 'Image_name', 'Maker', 'Genmodel', 'Year', 'Color', 'Genmodel_ID', 'Instance', 'class_index'])

            # Save dataframe
            info.to_csv(csv)

        return(info)

if __name__ == '__main__':

    # Instantiate dataset
    dvmcar = DvmCarDataset()

    # Print length
    print('DvmCarDataset length: {0}'.format(len(dvmcar)))