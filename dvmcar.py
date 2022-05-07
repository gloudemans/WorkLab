#from xmlrpc.client import Boolean
import torch
from torch.utils.data import Dataset

import os
import gdown
import glob

#from torchvision import datasets
#from torchvision.transforms import ToTensor
#import matplotlib.pyplot as plt
import pandas as pd
from torchvision.io import read_image
from zipfile import ZipFile

# Torch wrapper for DVM-CAR dataset.
# Static method fetch
class DvmCarDataset(Dataset):

    def __init__(self, 
            #annotations_file, 
            #img_dir, 
            #transform=None, 
            #target_transform=None,
            work:str=None, 
            persist:str=None, 
            url:str=None):  

        # If work not specified...
        if not work:

            # Set default work path
            work ='/data/dvmcar/dvmcar.zip'

        # If url not specified...
        if not url:

            # Set default download url
            url = 'https://figshare.com/ndownloader/articles/19586296/versions/1'

        # Fetch data to work file
        DvmCarDataset.fetch_zip(work, persist, url)

        # Unpack work file
        DvmCarDataset.unpack_zip(work)

        # Store work directory
        self.work_dir = os.path.split(work)[0]

        # Load useful data tables as pandas datframes
        # self.ad_df    = pd.read_csv(os.path.join(self.work_dir, 'Ad_table.csv'), skipinitialspace=True)
        self.basic_df = pd.read_csv(os.path.join(self.work_dir, 'Basic_table.csv'), skipinitialspace=True)
        self.image_df = pd.read_csv(os.path.join(self.work_dir, 'Image_table.csv'), skipinitialspace=True)
        # self.price_df = pd.read_csv(os.path.join(self.work_dir, 'Price_table.csv'), skipinitialspace=True)
        # self.sales_df = pd.read_csv(os.path.join(self.work_dir, 'Sales_table.csv'), skipinitialspace=True)
        # self.trim_df  = pd.read_csv(os.path.join(self.work_dir, 'Trim_table.csv'), skipinitialspace=True)

        # Get image info as a pandas dataframe
        self.info_df = DvmCarDataset.load_image_info(work, ['resized_DVM', 'confirmed_fronts'])

        self.image_df.set_index('Image_name')

        #self.info_df.set_index('Image_name')

        #left3 = left2.set_index('keyLeft')
        #left3.merge(right2, left_index=True, right_on='keyRight')

        self.info_df = self.info_df.merge(self.image_df, left_index=True, on='Image_name')

        print(self.info_df.columns)

    def __len__(self):

        # Number of rows in the dataframe
        return len(self.info_df.index)

    def __getitem__(self, idx):

        # Retrieve path to image
        img_path = self.info_df['filepath'][idx]

        # Read the image
        image = read_image(img_path)

        # Get the label number
        label = self.info_df['labelnum'][idx]

        # if self.transform:
        #     image = self.transform(image)
        
        # if self.target_transform:
        #     label = self.target_transform(label)
        
        return image, label   

    @classmethod
    def fetch_zip(cls, work:str, persist:str, url:str):

        try:

            print('Trying...')

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
                print('Copied work file from {0} to {1}.'.format(persist, work))

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
                print('Copied work file from {0} to {1}.'.format(work, persist))

                # Nothing more to do   
                return True

        except:

            # Log progress 
            print('Failed to fetch {0}.'.format(work))

            # No joy
            return False

    @classmethod
    def unpack_zip(cls, work):

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
    def load_image_info(cls, work, dirs):

        # Get work directory
        work_dir = os.path.split(work)[0]

        # Form csv file path
        csv = os.path.join(work_dir, 'info.csv')

        # If csv file exists...
        if os.path.exists(csv):

            # Read info from csv
            info = pd.read_csv(csv)

        else:

            files = []
            for d in dirs:
                full = os.path.join(work_dir, d)
                files += glob.glob(full + '/**/*.jpg', recursive=True)

            # Nissan$$Pulsar$$2016$$Grey$$64_32$$457$$image_10.jpg

            fields = []
            labels = set()
            for f in files:
                fx = os.path.split(f)
                frags = fx[1].split('$$')
                fields.append([f] + [fx[1]] + frags)
                labels.add(frags[4])

            unique = list(labels)
            unique.sort()

            for f in fields:
                label = f[6]
                labelnum = unique.index(label)
                f.append(labelnum)

            # Create dataframe
            info = pd.DataFrame(fields, columns=['filepath','Image_name','maker','model','year','color','label','instance','filename','labelnum'])

            # Save dataframe
            info.to_csv(csv)

        return(info)                

a = DvmCarDataset()
print(a[50])


#DvmCarDataset.fetch_zip()

# from os.path import exists
# from os.path import join
# from os import makedirs
# import shutil

# import csv
# import random
# from zipfile import ZipFile

# class CarTools:
    
#     # Persistent dataset storage location
#     persist_root = 'work'
    
#     # Working dataset location
#     work_root = 'work'
    
#     # Link from which dvmcar dataset can be downloaded 
#     dvmcar_url = 'https://figshare.com/ndownloader/articles/19586296/versions/1'
#     dvmcar_dirname = 'dvmcar'    
#     dvmcar_zipname = 'dvmcar.zip'
#     dvmcar_tablename = 'tables_V2.0.zip'
#     dvmcar_imagenames = ['Confirmed_fronts.zip', 'resized_DVM_v2.zip']
    
#     # Link from which boxcars3d dataset can be downloaded
#     # https://drive.google.com/file/d/19LHLOmmVyUS1R4ypwByfrV8KQWnz2GDT/view?usp=sharing
#     boxcars3d_url = 'https://drive.google.com/uc?id=19LHLOmmVyUS1R4ypwByfrV8KQWnz2GDT'
#     boxcars3d_dirname = 'boxcars3d'
#     boxcars3d_zipname = 'boxcars3d.zip'
#     boxcars3d_pklnames = ['atlas.pkl', 'dataset.pkl', 'classification_splits.pkl', 'verification_splits.pkl']

#     @classmethod
#     def retrieve_dvmcar(cls):
        
#         # Use common retrival
#         cls.retrieve_zip(cls.dvmcar_dirname, cls.dvmcar_zipname, cls.dvmcar_url)
        
#     @classmethod
#     def retrieve_boxcars3d(cls):
        
#         # Use common retrival
#         cls.retrieve_zip(cls.boxcars3d_dirname, cls.boxcars3d_zipname, cls.boxcars3d_url)
        

        
#     @classmethod
#     def unzip_dvmcar(cls):
        
#         # Path to directory
#         path = join(cls.work_root, cls.dvmcar_dirname)
        
#         # Path to source file
#         zip_name = join(path, cls.dvmcar_zipname)
        
#         # Opening the zip file
#         with ZipFile(zip_name, 'r') as zipobj:
            
#             # Get file names
#             file_names = zipobj.namelist()
            
#             # For each file name...
#             for file_name in file_names:

#                 # Create full file name
#                 full = join(path, file_name)
                
#                 # If zip file doesn't exist...
#                 if not exists(full):
                    
#                     # Extract it
#                     zipobj.extract(file_name, path)
                    
#                     # Extract its contents
#                     with ZipFile(full) as inner_zip:

#                         # Extract its contents
#                         inner_zip.extractall(path)                    

#     @classmethod
#     def unzip_boxcars3d(cls):
        
#         # Path to directory
#         path = join(cls.work_root, cls.boxcars3d_dirname)
        
#         # Path to source file
#         zip_name = join(path, cls.boxcars3d_zipname)
        
#         # Opening the zip file
#         with ZipFile(zip_name, 'r') as zipobj:

#             # For each pkl file...
#             for pkl in cls.boxcars3d_pklnames:
                
#                 file = "".join(('BoxCars116k/', pkl))
                
#                 print(file)

#                 # Extract contents
#                 zipobj.extract(file, path)
                                    
#     @classmethod
#     def dvmcar_make_model(cls):
        
#         # Path to directory
#         path = join(cls.work_root, cls.dvmcar_dirname)
        
#         # Path to tables file
#         table_zip_path = join(path, cls.dvmcar_tablename)
  
#         # Opening the tables zip file
#         with ZipFile(table_zip_path, 'r') as zip_obj:

#             # Dictionary to track unique instances by genmodel
#             genmodels = {}
            
#             # Open the image table
#             with zip_obj.open('Image_table.csv') as image_file:
                
#                  # Read the file and split it into lines
#                 lines = image_file.read().decode(encoding='utf-8').split("\n")
                
#                 # For each non-header row..
#                 for line in lines[1:]:
                    
#                     # Split the path name into fields
#                     fields = line.split(",")
                                        
#                     # Isolate genmodel and instance
#                     if len(fields) >= 2:
                    
#                         # Extract the genmodel id
#                         genmodel_id = fields[0]
                        
#                         # Extract the image id
#                         image_id = fields[1]

#                         # Split the image id into genmodel id, instance id, and image number fields
#                         imfields = image_id.split("$$")
                        
#                         # Get the instance id field
#                         instance_id = imfields[1]

#                         # If the key exists...
#                         if genmodel_id in genmodels:

#                             # Update the instance dictory
#                             genmodels[genmodel_id][instance_id] = None

#                         else:
                            
#                             # Start an instance dictionary
#                             genmodels[genmodel_id] = {instance_id : None}
                            
#                     else:
                        
#                         print("Table contained a line with too few fields: {0}".format(line))
                        
#             # Open the image table
#             with zip_obj.open('Basic_table.csv') as image_file:
                
#                  # Read the file and split it into lines
#                 lines = image_file.read().decode(encoding='utf-8').split("\n")
                
#                 basic = []
                
#                 # For each non-header row..
#                 for line in lines[1:]:
                    
#                     # Split the path name into fields
#                     fields = line.split(",")
                                        
#                     # Isolate genmodel and instance
#                     if len(fields)==4:

#                         # Add record
#                         basic.append(fields)
                        
#             def get_basic(genmodel_id, basic):
#                 for row in basic:
#                     if row[3]==genmodel_id:
#                         return row
#                 return None
                                              
#             instances = [[key, len(genmodels[key])] for key in genmodels.keys()]
            
#             automakers = {}
            
#             with open('InstancesByGenmodel.csv', 'wt') as csv_file:

#                 csv_file.write('Maker, Model, Instances\n')
#                 for genmodel_id, instances in instances:
#                     row = get_basic(genmodel_id, basic)
#                     automaker = row[0]
#                     model     = row[2]
#                     csv_file.write('{0},{1},{2}\n'.format(automaker, model, instances))
                    
#                     if automaker in automakers:
#                         automakers[automaker]+=instances
#                     else:
#                         automakers[automaker]=instances
                        
#             with open('InstancesByMaker.csv', 'wt') as csv_file:
#                 csv_file.write('Maker, Instances\n')
#                 for automaker, instances in automakers.items():
#                     csv_file.write('{0},{1}\n'.format(automaker, instances))
                    
                        
                
# # Retrieve datasets
# CarTools.retrieve_dvmcar()
# CarTools.retrieve_boxcars3d()
# CarTools.unzip_dvmcar()
# CarTools.unzip_boxcars3d()
# #CarTools.dvmcar_make_model()