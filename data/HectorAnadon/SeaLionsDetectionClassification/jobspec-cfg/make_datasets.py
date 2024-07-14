from keras.utils.io_utils import HDF5Matrix
from PIL import Image
import numpy as np
import os, sys
import time
import h5py


from usefulFunctions import *
from global_variables import *


def unison_shuffled_copies(a, b, c=None, d=None):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    if c != None:
        assert len(a) == len(c)
        if d != None:
            assert len(a) == len(d)
            return a[p], b[p], c[p], d[p]
        else:
            return a[p], b[p], c[p]
    else:
        return a[p], b[p]


def split_data(X_data, y_data, train_split):
    num_images = y_data.shape[0]
    X_train = X_data[0:int(round(train_split*num_images))]
    y_train = y_data[0:int(round(train_split*num_images))]
    X_test = X_data[int(round(train_split*num_images))+1:-1]
    y_test = y_data[int(round(train_split*num_images))+1:-1]
    return X_train, y_train, X_test, y_test


def save_to_disk(X, y, file_name):
    # Save to disk
    f = h5py.File(file_name, 'w')
    # Create dataset to store images
    X_dset = f.create_dataset('data', X.shape, dtype='f')
    X_dset[:] = X
    print (X_dset.shape)
    # Create dataset to store labels
    y_dset = f.create_dataset('labels', y.shape, dtype='i')
    y_dset[:] = y
    print (y_dset.shape)
    f.close()


def get_positive_samples(path, radius, net):
    """Get an array of positive samples (windows containing lions), their upper left corner 
    coordinates and their labels (both in binary and multiclass one-hot representation)
    """
    resolution_lvl = get_resolution_level(net)
    file_names = os.listdir(path + "Data/Train/")
    positive_samples = []
    # corners = []
    binary_labels = []
    multiclass_labels = []

    for image_name in file_names:
        if image_name.endswith('.jpg'):
            print("Processing ", image_name)
            image = Image.open(path + "Data/Train/" + image_name)
            #CHECK IF THERE IS A SEA LION IN THE IMAGE
            coordinates = extractCoordinates(path , image_name)
            classes = enumerate(CLASSES)
            for class_index, class_name in classes:
                for lion in range(len(coordinates[class_name][image_name])):
                    # Only consider sea lions within radius pixels from the edge
                    # TODO consider edge cases
                    x = coordinates[class_name][image_name][lion][0]
                    y = coordinates[class_name][image_name][lion][1]
                    if x > radius and x < (image.size[0] - radius) and y > radius and y < (image.size[1] - radius):
                        # Crop window of chosen resolution level
                        window = cropAndChangeResolution(image, image_name, x - radius, y - radius, radius * 2, resolution_lvl)
                        # Append
                        positive_samples.append(np.array(window))
                        # corners.append(np.array([x - radius, y - radius]))
                        if (net <= 3):
                            binary_labels.append([1,0])
                        else:
                            multiclass =  np.zeros(5, 'uint8')
                            multiclass[class_index] = 1
                            multiclass_labels.append(multiclass)
    # Concatenate
    positive_samples = np.float64(np.stack(positive_samples))
    #corners = np.uint16(np.stack(corners))
    if (net <= 3):
        binary_labels = np.uint8(np.array(binary_labels))
    else:
        multiclass_labels = np.uint8(np.stack(multiclass_labels))
    # Normalize data
    positive_samples /= 255.0
    # Save to disk
    f = h5py.File(path + 'Datasets/data_positive_net'+str(net)+'_small.h5', 'w')
    # Create dataset to store images
    X_dset = f.create_dataset('data', positive_samples.shape, dtype='f')
    X_dset[:] = positive_samples
    print(X_dset.shape)
    # Create dataset to store corners
    # corners_dset = f.create_dataset('corners', corners.shape, dtype='i')
    # corners_dset[:] = corners
    # Create dataset to store labels
    if (net <= 3):
        y_dset = f.create_dataset('labels', binary_labels.shape, dtype='i')
        y_dset[:] = binary_labels
        print (y_dset.shape)
    else:
        y_dset = f.create_dataset('labels', multiclass_labels.shape, dtype='i')
        y_dset[:] = multiclass_labels  
        print (y_dset.shape)
    f.close()
    return positive_samples.shape

def get_negative_samples(path, radius):
    """Get an array of negative samples (windows without lions), their upper left corner 
    coordinates and their labels (only binary format - NO SEA LION [0,1] / SEA LION [1,0])
    """
    file_names = os.listdir(path + "Data/Train/")
    res1 = get_resolution_level(1)
    res2 = get_resolution_level(2)
    res3 = get_resolution_level(3)
    negative_samples_net1 = []
    negative_samples_net2 = []
    negative_samples_net3 = []
    labels = []

    for image_name in file_names:
        if image_name.endswith('.jpg'):
            print("Processing ", image_name)
            image = Image.open(path + "Data/Train/" + image_name)
            coordinates = extractCoordinates(path , image_name)
            for it in range(NUM_NEG_SAMPLES):
                # Upper left corner coordinates
                x = np.random.uniform(0, image.size[0] - 2 * radius)
                y = np.random.uniform(0, image.size[1] - 2 * radius)
                window1 = cropAndChangeResolution(image, image_name, x, y, radius * 2, res1)
                window2 = cropAndChangeResolution(image, image_name, x, y, radius * 2, res2)
                window3 = cropAndChangeResolution(image, image_name, x, y, radius * 2, res3)
                label = getLabel(image_name, coordinates, x, y, radius * 2)
                # Append negative samples 
                if label == [0, 1]:
                    negative_samples_net1.append(np.array(window1))
                    negative_samples_net2.append(np.array(window2))
                    negative_samples_net3.append(np.array(window3))                    
                    labels.append(label)
    # Concatenate
    negative_samples_net1 = np.float64(np.stack(negative_samples_net1))
    negative_samples_net2 = np.float64(np.stack(negative_samples_net2))
    negative_samples_net3 = np.float64(np.stack(negative_samples_net3))
    labels = np.uint8(np.array(labels))
    # Normalize data
    negative_samples_net1 /= 255.0
    negative_samples_net2 /= 255.0
    negative_samples_net3 /= 255.0
    # Save to disk
    save_to_disk(negative_samples_net1, labels, path+'Datasets/data_negative_net1_small.h5')
    save_to_disk(negative_samples_net2, labels, path+'Datasets/data_negative_net2_small.h5')
    save_to_disk(negative_samples_net3, labels, path+'Datasets/data_negative_net3_small.h5')

    return negative_samples_net1.shape, labels.shape


def create_binary_net_datasets():
    """Combine positive and negative samples into one dataset (for each binary net).
    """
    # Load positive samples
    pos_samples1 = HDF5Matrix(PATH + 'Datasets/data_positive_net1_small.h5', 'data')
    pos_samples2 = HDF5Matrix(PATH + 'Datasets/data_positive_net2_small.h5', 'data')
    pos_samples3 = HDF5Matrix(PATH + 'Datasets/data_positive_net3_small.h5', 'data')
    pos_labels1 = HDF5Matrix(PATH + 'Datasets/data_positive_net1_small.h5', 'labels')
    pos_labels2 = HDF5Matrix(PATH + 'Datasets/data_positive_net2_small.h5', 'labels')
    pos_labels3 = HDF5Matrix(PATH + 'Datasets/data_positive_net3_small.h5', 'labels')
    # Check that labels are the same
    assert np.array_equal(pos_labels1, pos_labels2) and np.array_equal(pos_labels2, pos_labels3)
    pos_labels = pos_labels1
    # Load negative samples
    neg_samples1 = HDF5Matrix(PATH + 'Datasets/data_negative_net1_small.h5', 'data')
    neg_samples2 = HDF5Matrix(PATH + 'Datasets/data_negative_net2_small.h5', 'data')
    neg_samples3 = HDF5Matrix(PATH + 'Datasets/data_negative_net3_small.h5', 'data')
    neg_labels1 = HDF5Matrix(PATH + 'Datasets/data_negative_net1_small.h5', 'labels')
    neg_labels2 = HDF5Matrix(PATH + 'Datasets/data_negative_net2_small.h5', 'labels')
    neg_labels3 = HDF5Matrix(PATH + 'Datasets/data_negative_net3_small.h5', 'labels')
    # Check that labels are the same
    assert np.array_equal(neg_labels1, neg_labels2) and np.array_equal(neg_labels2, neg_labels3)
    neg_labels = neg_labels1
    # Check normalization
    assert np.amax(pos_samples1) <= 1 and np.amax(neg_samples1) <= 1
    assert np.amax(pos_samples2) <= 1 and np.amax(neg_samples2) <= 1
    assert np.amax(pos_samples3) <= 1 and np.amax(neg_samples3) <= 1
    # Concatenate positive and negative
    X1 = np.concatenate((pos_samples1, neg_samples1))
    X2 = np.concatenate((pos_samples2, neg_samples2))
    X3 = np.concatenate((pos_samples3, neg_samples3))
    y = np.concatenate((pos_labels, neg_labels))
    # Shuffle data
    X1, X2, X3, y = unison_shuffled_copies(X1, X2, X3, y)
    # Save to disk
    save_to_disk(X1, y, PATH + 'Datasets/data_net1_small.h5')
    save_to_disk(X2, y, PATH + 'Datasets/data_net2_small.h5')
    save_to_disk(X3, y, PATH + 'Datasets/data_net3_small.h5')
    return X1.shape, y.shape


def get_shifted_windows(image, image_name, x, y, resolution_lvl):

    # Offset vectors
    x_n = np.array(X_N)
    y_n = np.array(Y_N)
    corners = []

    windows = []
    labels = []
    num_transf = x_n.size * y_n.size

    window_size = getImageSize(resolution_lvl)

    transf = 0
    for delta_x in x_n:
        for delta_y in y_n:
            window = cropAndChangeResolution(image, image_name, x+delta_x, y+delta_y, window_size, resolution_lvl)
            windows.append(np.array(window))
            corners.append(np.array([x+delta_x, y+delta_y]))
            label = np.zeros(num_transf, 'uint8')
            label[transf] = 1
            labels.append(label)
            transf += 1
    return  np.stack(windows), np.stack(labels), np.uint16(np.stack(corners))

def get_calib_samples(path, radius, net):
    resolution_lvl = get_resolution_level(net)
    file_names = os.listdir(path + "Data/Train/")
    positive_samples = []
    corners = []
    labels = []

    for image_name in file_names:
        if image_name.endswith('.jpg'):
            print ("Processing ", image_name)
            image = Image.open(path + "Data/Train/" + image_name)
            #CHECK IF THERE IS A SEA LION IN THE IMAGE
            coordinates = extractCoordinates(path, image_name)
            classes = enumerate(CLASSES)
            for class_index, class_name in classes:
                for lion in range(len(coordinates[class_name][image_name])):
                    # Only consider sea lions within radius pixels from the edge
                    # TODO consider edge cases
                    x = coordinates[class_name][image_name][lion][0]
                    y = coordinates[class_name][image_name][lion][1]
                    if x > radius and x < (image.size[0] - radius) and y > radius and y < (image.size[1] - radius):
                        top_x = x - radius
                        top_y = y - radius
                        data, label, corner = get_shifted_windows(image, image_name, top_x, top_y, resolution_lvl)
                        positive_samples.append(data)
                        labels.append(label)
                        corners.append(corner)
    # Concatenate
    positive_samples = np.float64(np.concatenate(positive_samples))
    labels = np.uint8(np.concatenate(labels))
    corners = np.uint16(np.concatenate(corners))
    return positive_samples, labels, corners

def create_calib_dataset(path, window_size, net):
    import h5py
    radius = round(window_size / 2)
    # Get positive samples
    X, y, corners = get_calib_samples(path, radius, net)
    # Shuffle data
    X, corners, y = unison_shuffled_copies(X, corners, y)
    #Normalize
    X /= 255.0
    # Save to disk
    f = h5py.File(path + 'Datasets/data_calib'+str(net)+'_small.h5', 'w')
    # Create dataset to store images
    X_dset = f.create_dataset('data', X.shape, dtype='f')
    X_dset[:] = X
    # Create dataset to store corners
    corners_dset = f.create_dataset('corners', corners.shape, dtype='i')
    corners_dset[:] = corners
    # Create dataset to store labels
    y_dset = f.create_dataset('labels', y.shape, dtype='i')
    y_dset[:] = y
    f.close()
    return X.shape, y.shape, corners.shape


if __name__ == '__main__':

    """Create binary net datasets"""

    try:
        arg1 = sys.argv[1]
    except IndexError:
        print("Command line argument missing. Usage: make_datasets.py <mode>")
        sys.exit(1)

    if arg1 == 'pos1':
        print("POS 1\n", get_positive_samples(PATH, ORIGINAL_WINDOW_DIM / 2, 1))

    elif arg1 == 'pos2':
        print("POS 2\n", get_positive_samples(PATH, ORIGINAL_WINDOW_DIM / 2, 2))

    elif arg1 == 'pos3':
        print("POS 3\n", get_positive_samples(PATH, ORIGINAL_WINDOW_DIM / 2, 3))

    elif arg1 == 'neg':
        print("NEG\n", get_negative_samples(PATH, ORIGINAL_WINDOW_DIM / 2))

    elif arg1 == 'combine':
        print("COMBINED\n", create_binary_net_datasets())

    elif arg1 == 'multi':
        print("POS 4\n", get_positive_samples(PATH, ORIGINAL_WINDOW_DIM / 2, 4))

    elif arg1 == 'cal1':
        print("CALIB 1\n", create_calib_dataset(PATH, ORIGINAL_WINDOW_DIM, 1))

    elif arg1 == 'cal2':
        print("CALIB 2\n", create_calib_dataset(PATH, ORIGINAL_WINDOW_DIM, 2))

    elif arg1 == 'cal3':
        print("CALIB 3\n", create_calib_dataset(PATH, ORIGINAL_WINDOW_DIM, 3))

    else:
        print("Wrong command line argument.")



