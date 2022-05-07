import os
import gdown
import glob
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
from zipfile import ZipFile


class DvmCarDataset(Dataset):
    """PyTorch wrapper for DVM-CAR dataset.

    Args:
        Dataset (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Default work directory
    work_def = '/data/dvmcar/dvmcar.zip'

    # Default download url
    url_def = 'https://figshare.com/ndownloader/articles/19586296/versions/1'

    def __init__(self,
                 transform=None,
                 work: str = work_def,
                 persist: str = None,
                 url: str = url_def):
        """Construct a DvmCarDataset using specifies parameters. If dvmcar.zip 
        does not exist at the work location, the constructor attempt to copy it
        from persist and if that is unsuccessful downloads it from url to both
        work and persist. The constructor then unpacks the three subordinate zip
        files from dvmcar.zip and expands each of them. Two of the zip files contain
        images and the third contains supporting csv files. The constructor forms
        a pandas dataframe with the full paths to every image and various other fields extracted
        from the image filename. It joins this with the image table in order to 
        provide means of navigating from image id to path (though in practice we
        may not require that linkage). As the original class labels ("Genmodel ID")
        are strings, the constructor forms integer ids based on alphabeticaly ordering of
        unique labels and adds a column with these integer labels to the table.

        The getitem method optionally applies a transform to the image.

        Args:
            transform (_type_, optional): specifies getitem transform. Defaults to None.
            work (str, optional): specifies path to dvmcar.zip. Defaults to work_def.
            persist (str, optional): specifies path to persistent storage of dvmcar.zip. Defaults to None.
            url (str, optional): specifies downlaod url for dvmcar.zip. Defaults to url_def.
        """

        # Retain arguments
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
        # self.basic_df = pd.read_csv(os.path.join(
        #     self.work_dir, 'Basic_table.csv'), skipinitialspace=True)
        self.image_df = pd.read_csv(os.path.join(
            self.work_dir, 'Image_table.csv'), skipinitialspace=True)
        # self.price_df = pd.read_csv(os.path.join(
        #     self.work_dir, 'Price_table.csv'), skipinitialspace=True)
        # self.sales_df = pd.read_csv(os.path.join(
        #     self.work_dir, 'Sales_table.csv'), skipinitialspace=True)
        # self.trim_df = pd.read_csv(os.path.join(
        #     self.work_dir, 'Trim_table.csv'), skipinitialspace=True)

        # Get image info as a pandas dataframe
        self.info_df = DvmCarDataset.load_image_info(
            work, ['resized_DVM', 'confirmed_fronts'])

        # Drop duplicate images
        self.info_df.drop_duplicates(subset=['Image_name'], inplace=True)

        # # Join info with image table for viewpoint, quality
        self.info_df = pd.merge(self.info_df, self.image_df, left_on='Image_name', right_on='Image_name')

        # Sort on image name to be sure of order
        self.info_df.sort_values(by='Image_name', inplace=True)

    def __len__(self):
        """Returns number of images in dataset.

        Returns:
            _type_: Image count.
        """

        # Number of rows in the dataframe
        return len(self.info_df.index)

    def __getitem__(self, index : int):
        """Retrieve the indexed record from the dataset.

        Args:
            idx (int): specifies the item index

        Returns:
            (image, label): returns image and corresponding label 
        """

        # Retrieve path to image
        img_path = self.info_df['Image_path'][index]

        # Read the image
        image = read_image(img_path)

        # Get the label number
        label = self.info_df['Genmodel_number'][index]

        # If transform specified...
        if self.transform:

            # Apply transform
            image = self.transform(image)

        # Return image and label
        return image, label

    @classmethod
    def fetch_zip(cls, work: str, persist: str, url: str):
        """If the zip file specified by work does not exists, attempt to
        copy it from persist. If that doesn;t work attempt to download it
        from url to work and optionally copy it to persist. Return true 
        if the file exists at work on exit.

        Args:
            work (str): specifies full path to dvmcar.zip
            persist (str): optionally specifies full path to persistent copy of dvmcar.zip
            url (str): specifies url from which to download dvmcar.zip in liue of work and persist 

        Returns:
            Boolean: True if work has dvmcar.zip on exit
        """

        try:

            # Split work path
            work_dir = os.path.split(work)[0]

            # Coerce work directory
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
                os.shutil.copyfile(persist, work)

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

                # Coerce work directory
                os.makedirs(persist_dir, exist_ok=True)

                # Copy from work to persist
                os.shutil.copyfile(work, persist)

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
    def load_image_info(cls, work: str, dirs: list):
        """If info.csv exists in same directory as dvmcar.zip specified by work,
        load it. If not, iterate over directories specifies in dirs finding all jpg
        files recursively. Extract fields from the name of each jpg file.

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

                # Form a row containing the full path to jpg, the name of the jpg, and the fields extracted from the name
                records.append([jpg_path] + [jpg_name] + fields)

                # Add the "Genmodel_ID" field to to the list of class labels 
                genmodel_ids.add(fields[4])

            # Make a list out of the label set
            genmodel_ids = list(genmodel_ids)

            # Sort the list
            genmodel_ids.sort()

            # For each record...
            for record in records:

                # Get its genmodel id
                genmodel_id = record[6]

                # Convert to integer index
                genmodel_number = genmodel_ids.index(genmodel_id)

                # Append integer genmodel
                record.append(genmodel_number)

            # Create dataframe
            info = pd.DataFrame(records, columns=[
                                'Image_path', 'Image_name', 'Maker', 'Genmodel', 'Year', 'Color', 'Genmodel_ID', 'Instance', 'Image_tail', 'Genmodel_number'])

            # Save dataframe
            info.to_csv(csv)

        return(info)

if __name__ == '__main__':
    # Running as a script

    dvmcar = DvmCarDataset()
    print(dvmcar.info_df[['Genmodel','Year','Image_name','Genmodel_number']][:5])

