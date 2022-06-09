import os
from os.path import exists
from os.path import join
import gdown
import shutil
import json
import torch
import math
import numpy as np
from torch import classes, nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda
from torchvision.io import read_image
import pandas as pd
from zipfile import ZipFile

import torchvision

import matplotlib.pyplot as plt

class BoxCars3dDataset(Dataset):
    """PyTorch wrapper for BoxCars3D dataset.

    Args:
        Dataset (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Default local work directory
    work_def = '/data/boxcars3d/boxcars3d.zip'   

    # Default download url
    url_def = 'https://drive.google.com/uc?id=19LHLOmmVyUS1R4ypwByfrV8KQWnz2GDT'

    def __init__(self,
                work: str = work_def,
                persist: str = None,
                url: str = url_def,
                split = [0, 1],
                transform = None,
                label_type = 'label'):

        # Retain arguments
        self.work = work
        self.persist = persist
        self.url = url
        self.split = split
        self.transform = transform
        self.label_type = label_type

        # Fetch data to work file
        BoxCars3dDataset.fetch_zip(work, persist, url)

        # Unpack work file
        BoxCars3dDataset.unpack_zip(work)

        # Store work directory
        self.work_dir = os.path.split(work)[0]

        # Get path to dataset json file
        json_name = os.path.join(self.work_dir, 'BoxCars116k/json_data/dataset.json')

        # Load dataset json file
        with open(json_name, "r") as json_file:

            # Load data from the json file
            json_data = json.load(json_file)

            # Get list of unique class names
            labels = list(set([s['annotation'] for s in json_data['samples']]))
            labels.sort()
            self.label_list = labels
            self.label_dict = dict(zip(labels, range(len(labels))))

            # Labels have maker, model, body and mk fields
            label_fields = [l.split() for l in labels]

            # Get sorted unique makers
            makers = list(set([l[0] for l in label_fields]))
            makers.sort()
            self.maker_list = makers
            self.maker_dict = dict(zip(makers, range(len(makers))))

            # Get sorted unique models
            models = list(set([l[1] for l in label_fields]))
            models.sort()
            self.model_list = models
            self.model_dict = dict(zip(models, range(len(models))))

            # Get sorted unique body types
            bodys = list(set([l[2] for l in label_fields]))
            bodys.sort()
            self.body_list = bodys
            self.body_dict = dict(zip(bodys, range(len(bodys))))

            # Get sorted unique "mks" - maybe trim variants?
            mks = list(set([l[3] for l in label_fields]))
            mks.sort()
            self.mk_list = mks
            self.mk_dict = dict(zip(mks, range(len(mks))))

            # Make empty list
            dataframe = []

            # For each sample...
            for s in json_data['samples']:

                # Retrieve the sample annotion field
                label = s['annotation']

                # Get the corresponding integer class index
                label_n = self.label_dict[label]

                # Split the label test into fields
                fields = label.split()

                # Unpack fields extracted from the label
                maker, model, body, mk = fields

                # Get integer indices corresponding to maker, model, body, and "mk"
                maker_n = self.maker_dict[maker]
                model_n = self.model_dict[model]
                body_n = self.body_dict[body]
                mk_n = self.mk_dict[mk]

                # Retrieve the sample instances field
                instances = s['instances']

                # For each instance...
                for i in instances:

                    # Append a list including various integer class labels and the image path
                    dataframe.append([label_n, maker_n, model_n, body_n, mk_n, i["path"]])

            # Convert the list of lists into a pandas dataframe
            self.dataframe = pd.DataFrame(dataframe, columns =['Label_N', 'Maker_N', 'Model_N', 'Body_N', 'Mk_N', 'Image_Path'])

         # Sort on image name to be sure of order
        self.dataframe.sort_values(by='Image_Path', inplace=True)

        # Create a random number generator with repeatable seed
        rng = np.random.default_rng(seed=0)

        # Create shuffled order for data splits
        self.shuffle = rng.permutation(len(self.dataframe))

        # Set the specified dataset split
        self.set_split(split)

    def __len__(self):
        """Returns the number of images in the currently defined
        dataset split.

        Returns:
            _type_: Image count.
        """

        # Number of rows in the dataframe split
        return self.i1-self.i0

    def set_label_type(self, label_type : str):
        """Set the label type to return with getitem. The 
        dataset can provide the following label types:
          'label' - fine class including maker, model, body, and mk  
          'maker' - vehicle maker
          'model' - vehicle model
          'body'  - vehicle body type

        Args:
            label_type (str): specifies the label type
              
        """ 
        self.label_type = label_type

    def __getitem__(self, index: int):
        """Retrieve the indexed record from the dataset. 
        Return value is a tuple including the transformed image and
        corresponding label based on the currently specified label 
        type. 

        Args:
            idx (int): specifies the item index

        Returns:
            (image, label): returns image and corresponding label 
        """

        # Randomly permute the index
        shuffled_index = self.shuffle[index+self.i0]

        # Retrieve path to image
        img_path = os.path.join(self.work_dir, 'BoxCars116k/images', self.dataframe['Image_Path'][shuffled_index])

        # Read the image
        image = read_image(img_path).float()

        if self.label_type=='label':

            # Get the label number
            label = self.dataframe['Label_N'][shuffled_index]
            
        elif self.label_type=='maker':

            # Get the label number
            label = self.dataframe['Maker_N'][shuffled_index]

        elif self.label_type=='model':

            # Get the label number
            label = self.dataframe['Model_N'][shuffled_index]

        elif self.label_type=='body':

            # Get the label number
            label = self.dataframe['Body_N'][shuffled_index]

        # If transform specified...
        if self.transform:

            # Apply transform
            image = self.transform(image)            

        # Return image and label
        return image, label

    def set_split(self, split:list):
        """Defines a subset of samples to be returned by getitem.
        Split should define a range as a pair of nundecreasing 
        real numbers ranging from 0 to 1. The split [0,1] includes
        the entire dataset, while [0,.5] includes the lower half, etc.

        Args:
            split (list): _description_
        """
        # Save split 
        self.split = split

        # Set split limits
        self.i0 = math.floor(split[0]*len(self.dataframe))
        self.i1 = math.floor(split[1]*len(self.dataframe))           

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

        # Split work file path
        work_dir = os.path.split(work)[0]
        
        # Opening the zip file
        with ZipFile(work, 'r') as work_zip:

            # Get file list
            names = work_zip.namelist()

            # For each pkl file...
            for f in names:
                
                path = os.path.join(work_dir, f)
                #print(path)

                if not exists(path):

                    # Extract contents
                    work_zip.extract(f, work_dir)            

if __name__ == '__main__':

    # Instantiate dataset
    boxcars3d = BoxCars3dDataset()

    # Print length
    print('BoxCars3dDataset length: {0}'.format(len(boxcars3d)))

    image, label = boxcars3d[0]
    n = 5

    for k in range(n):
        image, label = boxcars3d[k]
        print(image.shape)
        plt.imshow(image.permute(1, 2, 0)/256)
        plt.title(boxcars3d.label_list[label])
        plt.show()
