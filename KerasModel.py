# CSC 3520 Homework 3
# Jake Andersen and Christopher Villegas
# Uses a Keras model with 32-16-8-128-5-1 architecture to predict heart disease from input data.
# Each input sample has 11 features:
# Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
# Data retrieved from https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import pdb
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import keras
from keras import models
import random
import tensorflow as tf

ROOT = os.path.dirname(__file__)


parser = argparse.ArgumentParser(
    description='Use a Neural Network to predict heart failure')
parser.add_argument('-d', '--data', help='Path to data file, defaults to ROOT/data.csv',
                    default=os.path.join(ROOT, 'data.csv'))
parser.add_argument('-n', '--name', type=str, help='Name of the network')
parser.add_argument('-e', '--epochs', type=int,
                    help='Number of epochs to train for, defaults to 200', default=200)
parser.add_argument('-r', '--rate', type=float,
                    help='Learning rate (eta) of the network, defaults to 0.01', default=0.01)
parser.add_argument('-p', '--patience', type=int,
                     help="Number of epochs with no significant change before stopping, defaults to 20", default=20)
parser.add_argument('-s', '--seed', type=int, help='Set the seed for reproducibility')
parser.add_argument('-v', '--verbose', type=int,choices=[0,1,2], help='Prints information [0 = No information, 1 = Default information for keras, 2 = No progress bars]', default=1)


def main(args):

    if args.seed:
        # Set random seeds for reproducibility
        np.random.seed(args.seed)
        random.seed(args.seed)
        tf.random.set_seed(args.seed)


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


    # Build the Model
    num_features = xtrain.shape[1]
    model = models.Sequential()
    model.add(keras.Input(shape=(num_features,), name=args.name))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(128, activation='sigmoid'))
    model.add(keras.layers.Dense(5, activation='sigmoid'))
    model.add(keras.layers.Dense(1, use_bias=False, activation='sigmoid'))


    if args.verbose:
        model.summary()
        input("Press Enter to train the model")


    # Compile the model
    model.compile(
        loss='mse',
        optimizer=keras.optimizers.Adamax(learning_rate=args.rate),
        metrics=['accuracy']
    )


    # Create the early stoppiing callback
    callback = keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=1e-3,
        patience=args.patience,
        verbose=1
    )


    # Fit the model to the training data
    history = model.fit(xtrain.astype(np.float32), ytrain.astype(np.float32), batch_size=len(xtrain), epochs=args.epochs, callbacks=callback, validation_data=(xval.astype(np.float32), yval.astype(np.float32)), verbose=int(args.verbose))


    print('\n***PERFORMANCE METRICS***')

    # Test the network
    pred = model.predict(xtest.astype(np.float32)) >= 0.5 # Obtain the array of predictions
    acc = np.mean(pred.T==ytest)
    print(f'Testing Accuracy:  {np.sum(pred.T==ytest)}/{len(ytest)}     ({acc*100:0.2f}%)')

    # Create Confusion Matrix
    cm = confusion_matrix(ytest,pred)

    # Calculate the Precision and Recall
    pre = cm[1, 1]/(cm[1, 1]+cm[0, 1])
    rec = cm[1, 1]/(cm[1, 1]+cm[1, 0])
    print(f"Testing Precision: {cm[1, 1]}/{cm[1, 1]+cm[0, 1]}     ({pre*100:.2f}%)")
    print(f"Testing Recall:    {cm[1, 1]}/{cm[1, 1]+cm[1, 0]}     ({rec*100:.2f}%)")

    
    # Plot History
    plt.figure(1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label = 'Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot the confusion matrix
    plot_matrix(cm)

    # Show both figures
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


if __name__ == "__main__":
    main(parser.parse_args())
