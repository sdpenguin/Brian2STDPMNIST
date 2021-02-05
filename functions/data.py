'''
A variety of functions for managing the loading of MNIST data.
'''

import os
import pickle
import numpy as np
from struct import unpack
from tqdm import tqdm

def get_labeled_data(picklename, bTrain = True, MNIST_data_path='./mnist'):
    """ Read input-vector (image) and target class (label, 0-9) and return
        it as list of tuples.
        picklename: Path to the output pickle file.
        bTrain: True if training data, else False for test data.
        MNIST_data_path: Directory containing the MNIST files.
    """
    if os.path.isfile('{}.pickle'.format(picklename)):
        data = pickle.load(open('{}.pickle'.format(picklename), mode='rb'))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(os.path.join(MNIST_data_path, 'train-images.idx3-ubyte'), mode='rb')
            labels = open(os.path.join(MNIST_data_path, 'train-labels.idx1-ubyte'), mode='rb')
        else:
            images = open(os.path.join(MNIST_data_path, 't10k-images.idx3-ubyte'), mode='rb')
            labels = open(os.path.join(MNIST_data_path, 't10k-labels.idx1-ubyte'), mode='rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]

        if number_of_images != N:
            raise ValueError('The number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        print('Unpacking {} images...'.format('training' if bTrain else 'test'))
        for i in tqdm(range(N)):
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]

        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("{}.pickle".format(picklename), "wb"))
    return data