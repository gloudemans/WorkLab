import time
import os
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import models, transforms


import matplotlib.pyplot as plt

from dvmcar import DvmCarDataset

# load_data - return dvmcar training and validation datasets
#
# locale - specifies environment in which code is exectuting since paths may need
#   to be modified.
#     "Lambda Labs"
#     "Default"
# scale - fraction of dataset to use, 1 uses the entire dataset

def load_data(locale="Default", scale=0.02):

    # Depending on the locale...
    if locale=="Lambda Labs":
        # Use lambda labs paths
        work_def = '/home/ubuntu/WorkLab/data/dvmcar/dvmcar.zip'
        persist_def = '/home/ubuntu/worklab/dvmcar.zip'
    else:
        # Use default paths
        work_def = '/data/dvmcar/dvmcar.zip'
        persist_def = None

    # Set partitions for train, test, and validate subsets
    partition0  = 0.8*scale
    partition1  = 0.9*scale
    partition2  = 1.0*scale

    # Define corresponding split arguments for the dataset constructor
    train_split = [0,          partition0]
    val_split   = [partition0, partition1]
    test_split  = [partition1, partition2]
    
    # Resnet input height & width?
    input_size  = 224

    # Specify training transform stack
    # Not too sure what random resize crop does...
    # Per Derek - maybe color space & other distortions would be useful?
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])

    # Specify validation transform stack
    val_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])

    train_data  = DvmCarDataset(split = train_split, transform = train_transform, work = work_def, persist = persist_def)
    val_data    = DvmCarDataset(split =   val_split, transform =   val_transform, work = work_def, persist = persist_def)
    
    return(train_data, val_data)

class stat_accumulator:

    def __init__(self, classes):
        self.classes = classes
        self.top = np.zeros(self.classes)
        self.count = 0
        
    def update(self, labels, outputs):
        
        # Get indices (labels) sorting the vector in decreasing order
        top_indices = torch.argsort(outputs, 1, descending=True)

        # Find predicted ranking of the true class for each result in the batch
        n = [(top_indices[k]==labels.data[k]).nonzero().squeeze().item() for k in range(len(labels.data))]

        # For each result in the batch...
        for idx in n:

            # Increment
            self.top[idx] += 1
            
        self.count += len(labels.data)
        
        return(self.top[0]/self.count, self.count, self.top/self.count)
        
class stat_plotter:
    
    def __init__(self, name):
        plt.ion()
        plt.rcParams['figure.dpi'] = 160

        self.name = name
        self.fig = plt.figure(num=name)
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2)

    # def __del__(self):
    #    plt.ioff()
    #    plt.show()
        
    def update(self, rank_count, phase, epoch, sample):
        
        # Update plot
        self.fig.suptitle('{}: epoch={:3}, sample={:8}'.format(
            self.name, epoch, sample))

        #clear_output(wait = True)
        x = np.arange(rank_count.size) + 1
        
        max_n = 20

        self.ax1.cla()
        self.ax1.plot(x,          rank_count , label='P(rank==N)')
        self.ax1.plot(x,np.cumsum(rank_count), label='P(rank in 1..N)')
        self.ax1.set_title('All classes train')
        self.ax1.legend()
        self.ax1.grid()
        self.ax1.set_ylabel('Probability')
        self.ax1.set_xlabel('N')    

        self.ax2.cla()
        self.ax2.plot(x[:max_n],          rank_count[:max_n] , label='P(rank==N)', linestyle='None', marker='o')
        self.ax2.plot(x[:max_n],np.cumsum(rank_count[:max_n]), label='P(rank in 1..N)', linestyle='None', marker='o')
        self.ax2.set_title('Top {} classes train'.format(max_n))
        self.ax2.legend()
        self.ax2.grid()
        self.ax2.set_ylabel('Probability')
        self.ax2.set_xlabel('N')    
        self.fig.tight_layout()      

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()       

        print('Epoch={:3d}, Phase={:5}, Sample={:8d}'.format(epoch, phase, sample)) 

class model_manager:
    
    def __init__(self, classes):
        
        # Input image height and width
        self.image_size = 224
        
        # Set number of output classes
        self.classes = classes
        
        # Set filename for shutdown file
        self.stop_file = 'stop.txt'
        
    def train(self, datasets, 
        checkpoint_file = "DvmCar.pt",
        weight_file = "DvmCar.wt",
        batch_size = 50,
        max_epochs = 100,
        max_hours = 24,
        checkpoint_minutes = 10,
        plot_samples = 1000,
        feature_extract = True,
        resume = False):
        
        # Zero out shutdown file at start of training
        with open(self.stop_file, 'w') as f:
            pass
        
        # Create training and validation dataloaders
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
        
        # Clear epoch
        epoch = 0
        
        # Clear loss
        loss = 0
        
        # Clear accuracy
        accuracy = 0
        
        # Clear best accuracy
        best_acc = 0
        
        # Record time at start of training
        training_start = time.time()       
        
        train_plotter = stat_plotter("Train")
        val_plotter = stat_plotter("Val")
        
        val_acc_history = []
        
        # Start the checkpoint timer
        checkpoint_timer = training_start
                
        # Start with standard resnet 50
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Replace fully connected layer with the required number of output classes
        # (These will have requires_grad = True)
        model.fc = nn.Linear(model.fc.in_features, self.classes)
        
        # Empty parameter list
        params_to_update = []

        # For each parameter in the list...
        for name, param in model.named_parameters():

            # Determine whether to update this parameter
            param.requires_grad = (not feature_extract) or ((name=="fc.weight") or (name=="fc.bias"))
            
            # If updating this parameter
            if param.requires_grad:

                # Append it to the parameter list
                params_to_update.append(param)

        # Create optimizer
        optimizer = optim.AdamW(params_to_update)         
        
        # Setup the loss function
        criterion = nn.CrossEntropyLoss()
        
        # If resuming training from checkpoint...
        if resume:
            
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            accuracy = checkpoint['accuracy']
            
        else:

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'accuracy': accuracy}, checkpoint_file) 
            
        print("cuda:0" if torch.cuda.is_available() else "cpu")

        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Send the model to GPU
        model = model.to(device)
                    
        # Clear completion flag
        training_complete = False
        
        # While training remains incomplete...
        while not training_complete:
        
            # Log epoch start
            # self.log_epoch_start(epoch)
            print("Epoch Start: {}".format(epoch))

            # For each phase...
            for phase in ['train', 'val']:
                
                # If training complete...
                if training_complete:
                    
                    # Stop looping
                    break

                # Depending on the phase
                if phase == 'train':
                    
                    # Set model to training mode
                    model.train()
                    
                else:
                    
                    # Set model to evaluate mode
                    model.eval()   

                # Clear accumulators
                running_loss = 0.0
                running_corrects = 0
                sample = 0
                
                # Start the plot timer
                plot_sample = 0
                
                # Create stat accumulator
                stats = stat_accumulator(self.classes)

                # For each minibatch...
                for inputs, labels in dataloaders[phase]:
                    
                    # If training complete...
                    if training_complete:

                        # Stop looping
                        break                    
                    
                    # Move inputs and labels to GPU
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Update sample counter
                    sample += inputs.size(0)

                    # Clear gradients
                    optimizer.zero_grad()

                    # forward
                    # track history only if training
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        # Get model outputs
                        outputs = model(inputs)
                        
                        # Calculate loss
                        loss = criterion(outputs, labels)

                        # Use maximal class activations as predictions
                        _, preds = torch.max(outputs, 1)
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            
                    # Update running statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    # Update stat accumulator
                    accuracy, count, top = stats.update(labels, outputs)
                    
                    # Get present time
                    now = time.time()

                    # If it is time to update plots...
                    if sample-plot_sample>= plot_samples:
                        
                        if phase == 'train':
                            
                            # Update the plots
                            train_plotter.update(top, phase, epoch, sample)  
                            
                        else:
                            # Update the plots
                            val_plotter.update(top, phase, epoch, sample)  
                    
                        # Adjust the timer
                        plot_sample += plot_samples
                        
                        # Open the stopping file
                        with open(self.stop_file) as f:
                            
                            # Read content
                            content = f.read()
                            
                            # If file has any content... 
                            if len(content):

                                # Stop training
                                training_complete = True
                                
                                # User stopped
                                completion_criteria = "User stopped"                               
                        
                    # If it is time to write a checkpoint...
                    if now - checkpoint_timer >= checkpoint_minutes*60:
                        
                        # Write the checkpoint
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            'accuracy': accuracy}, checkpoint_file) 
                        
                        # Reset checkpoint timer
                        checkpoint_timer = now
                    
                    # If elapsed training time exceeds limit...
                    if now - training_start >= max_hours*3600:
                        
                        # Training is complete
                        training_complete = True
                        
                        # Max time exceeded
                        completion_criteria = "Time limit reached"    
            
                    # Update epoch statistics
                    epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                    # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # If validation phase...
                if phase == 'val':
                
                    # Update eopoch accuracy list
                    val_acc_history.append(epoch_acc)
                    
                    # If validation performance improved...
                    if epoch_acc > best_acc:
                        
                        # Update best performance
                        best_acc = epoch_acc
                        
                        # Save best model weights
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'loss': loss,
                            'accuracy': accuracy}, weight_file)
                        
                        # Reset got worse count
                        got_worse = 0
                        
                    else:
                        
                        # Increment got worse count
                        got_worse += 1
                        
                        # If got worse too many times
                        if got_worse > 1:
                            
                            # Stop training
                            training_complete = True
                            
                            # Quit due to overfitting / decreasing accuracy
                            completion_criteria = "Accuracy got worse"
                 
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'accuracy': accuracy}, checkpoint_file) 
            
            # Increment epoch counter
            epoch += 1
            
            # If epoch limit reached...
            if epoch>=max_epochs:
                
                # Stop training
                training_complete = True
                
                # Quit due to epoch limit
                completion_criteria = "Reached epoch limit"
                
        print("Training complete")
        print("Completion criteria: {}".format(completion_criteria))            
        print("Epoch: {}".format(epoch))
        print("Accuracy: {}".format(accuracy))
        print("Best accuracy: {}".format(best_accuracy))

if __name__ == '__main__':

    train_data, val_data = load_data()

    print('Training split contains {:7} images.'.format(len(train_data)))
    print('Validation split contains {:7} images.'.format(len(val_data)))

    # Create training and validation datasets
    image_datasets = {'train': train_data, 'val' : val_data}

    manager = model_manager(train_data.classes)
    manager.train(image_datasets)
    # os.system("shutdown /s /t 1")