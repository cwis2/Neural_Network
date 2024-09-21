# CSC 3520 Homework 3
# Jake Andersen and Christopher Villegas
# Uses a Pytorch model with a 32-8-16-64-1 architecture to predict heart disease from input data.
# Each input sample has 11 features:
# Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope 
# Data retrieved from https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data


import argparse
import os
import pdb
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import random
import torch
from torch import nn


ROOT = os.path.dirname(__file__)


parser = argparse.ArgumentParser(
    description='Use a Neural Network to predict heart failure')
parser.add_argument('-d', '--data', help='Path to data file, defaults to ROOT/data.csv',
                    default=os.path.join(ROOT, 'data.csv'))
parser.add_argument('-n', '--name', type=str, help='Name of the network')
parser.add_argument('-e', '--epochs', type=int,
                    help='Number of epochs to train for, defaults to 10000', default=10000)
parser.add_argument('-r', '--rate', type=float,
                    help='Learning rate (eta) of the network, defaults to 0.001', default=0.001)
parser.add_argument('-s', '--seed', type=int, help='Set the seed for reproducibility')
parser.add_argument('-v', '--verbose',type=int, choices=[0,1], help='Prints information [0 = Disabled, 1 = Enabled]', default=1)


# Neural Network Class
class NeuralNetwork(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(num_features,32),
            nn.ReLU(),
            nn.Linear(32,8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16,64),
            nn.ReLU(),
            nn.Linear(64,1, bias=False),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.flatten(x)
        y = self.model(x)
        return y

def main(args):

    if args.seed:
        # Set random seeds for reproducibility
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)


    if args.verbose:
        print(f"\n***READING DATA FROM {args.data}***\n")

    # Read the data from the .csv file
    df = pd.read_csv(args.data)


    if args.verbose:
        print(f"Total samples read: {len(df)}\n")

        print("***PREPARING DATA***\n")


    # Find the non-numeric columns in the data to encode later
    cols = []
    for i in range(df.shape[1]):
        if not is_numeric_dtype(df.iloc[:,i].dtype):
            cols.append(i) 


    if args.verbose:
        print(f"Non-numeric columns to be encoded: {cols}\n")

    # Encode the data using one-hot encoding
    df = pd.get_dummies(df, columns=df.columns[cols])


    if args.verbose:
        print("***SPLITTING DATA INTO 70%/20%/10%***\n")

    # Splits the data into training (70% of total samples), validation (20% of total samples), and testing (10% of total samples) data
    train, validation, test = np.split(df.sample(frac=1, random_state=args.seed), [int(.7*len(df)), int(.9*len(df))])


    if args.verbose:
        print(f"\n# Training samples: {len(train)}, # Validation samples: {len(validation)}, # Testing samples: {len(test)}")

    # Will split the data into inputs and outputs
    # All elements in every row except the last column get saved into x all labels get saved into y
    xtrain, ytrain = train.iloc[:, :-1].values, train.iloc[:, -1].values
    xval, yval = validation.iloc[:, :-1].values, validation.iloc[:, -1].values
    xtest, ytest = test.iloc[:, :-1].values, test.iloc[:, -1].values

    # Turn data into tensors
    xtrain =  torch.tensor(xtrain.astype(np.float32), dtype=torch.float32)
    ytrain =  torch.tensor(ytrain.astype(np.float32), dtype=torch.float32)
    xval   =  torch.tensor(xval.astype(np.float32)  , dtype=torch.float32)
    yval   =  torch.tensor(yval.astype(np.float32)  , dtype=torch.float32)
    xtest  =  torch.tensor(xtest.astype(np.float32) , dtype=torch.float32)
    ytest  =  torch.tensor(ytest.astype(np.float32) , dtype=torch.float32)

    # Make the model 
    num_features = xtrain.shape[1]

    model = NeuralNetwork(num_features)

    if args.verbose:
        print(model)

        input("Press <Enter> to train this network...")

    # Create loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.rate)
    

    # Store loss history
    history = {"loss":[], 'val_loss':[]}
    for epoch in range(args.epochs):

        # Forward pass
        model.train()
        outputs = model(xtrain)
        loss = loss_fn(outputs.squeeze(), ytrain)  # Calculate the loss
        history['loss'].append(loss.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        val_out = model(xval)
        val_loss = loss_fn(val_out.squeeze(), yval)
        history['val_loss'].append(val_loss.item()) 

        # Print epoch data every 100 epochs
        if args.verbose and (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{args.epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

    # Test the network
    print("\n***PERFORMANCE METRICS***")

    # Creates a boolean array of predictions
    y = np.array([i[0] for i in model(xtrain) > 0.5]) 

    # Converts ytrain to a boolean array of values
    t = np.array(ytrain > 0)

    # Calculates training accuracy                          
    train_accuracy = np.mean(y == t)

    print(f"Training Accuracy: {np.sum(y==t)}/{len(t)}   ({train_accuracy*100:0.2f}%)")

    # Creates a boolean array of predictions
    y = np.array([i[0] for i in model(xtest) > 0.5])

    # Converts ytrain to a boolean array of values
    t = np.array(ytest > 0)

    # Calculates testing accuracy
    test_accuracy = np.mean(y == t)


    print(f"Testing Accuracy:  {np.sum(y==t)}/{len(t)}     ({test_accuracy*100:0.2f}%)")

    # Create confusion matrix
    cm = confusion_matrix(t, y)

    # Calculate precision and recall
    pre = cm[1, 1]/(cm[1, 1]+cm[0, 1])
    rec = cm[1, 1]/(cm[1, 1]+cm[1, 0])
    print(f"Testing Precision: {cm[1, 1]}/{cm[1, 1]+cm[0, 1]}     ({pre*100:.2f}%)")
    print(f"Testing Recall:    {cm[1, 1]}/{cm[1, 1]+cm[1, 0]}     ({rec*100:.2f}%)")

    # Plot History
    plt.figure(1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label = 'Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot the confusion matrix
    plot_matrix(cm)

    # Plot both figures
    plt.show()


def plot_matrix(cm, cmap=plt.cm.Greys_r, title="Confusion Matrix"):
    """
    Plot a confusion matrix.

    Parameters:
        cm (array-like): Confusion matrix.
        cmap (colormap, optional): The color map to use for the matrix. Defaults to plt.cm.Greys_r.
        title (str, optional): Title of the plot. Defaults to "Confusion Matrix".
    """
    # Define class labels
    classes = ["Healthy", "Heart Disease"]

    # Create a new figure
    plt.figure(2)

    # Plot the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    # Set the title
    plt.title(title)

    # Add color bar
    plt.colorbar()

    # Set x-axis ticks
    marks = range(2)
    plt.xticks(marks, classes, rotation=45)

    # Set y-axis ticks
    plt.yticks(marks, classes)

    # Set text format for cells
    fmt = 'd'
    thold = cm.max()/2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt), ha='center',
                        va='center', color='Black' if cm[i, j] > thold else 'White')

    # Adjust layout
    plt.tight_layout()

    # Set labels for axes
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")


if __name__=="__main__":
    main(parser.parse_args())
