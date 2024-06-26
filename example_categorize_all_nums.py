import numpy as np
import struct
import matplotlib.pyplot as plt

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    
def create_dataset(train_images_path, train_labels, test_images, test_labels):
    '''
    This function reads the image data from the files specified by the input
    arguments, and returns the training and testing data.
    '''
    # Read the images data
    train_images = read_idx(train_images_path)
    train_labels = read_idx(train_labels_path)
    test_images = read_idx(test_images_path)
    test_labels = read_idx(test_labels_path)
    
    return train_images, train_labels, test_images, test_labels

def create_part_of_all_dataset(train_images_path, train_labels, test_images, test_labels, percentage):
    '''
    This function reads the image data from the files specified by the input
    arguments, and returns the training and testing data.
    '''
    # Read the images data
    train_images = read_idx(train_images_path)
    train_labels = read_idx(train_labels_path)
    test_images = read_idx(test_images_path)
    test_labels = read_idx(test_labels_path)

    #take only a percentage of the train data and labels randomly
    if percentage < 1:
        train_images = train_images[:int(len(train_images)*percentage)]
        train_labels = train_labels[:int(len(train_labels)*percentage)]

    return train_images, train_labels, test_images, test_labels

def examples_categorize(train_images, train_labels, test_images, test_labels):
    '''
    This function calculates the accuracy of the examples-based categorization
    of the test images.
    The function calculates the distance between each test image and each
    training image, and assigns the label of the closest training image to
    the test image.
    The function returns the accuracy of the categorization.
    '''
    # Calculate the closest image in the training set for each image in the test set
    images_score = []
    len_test = len(test_images)
    labled_images_counter = 0
    for test_image in test_images:
        best_distance = np.inf
        image_label = None
        for i, train_image in enumerate(train_images):
            distance = np.mean((test_image - train_image) ** 2)
            if distance < best_distance:
                best_distance = distance
                image_label = train_labels[i]     
        images_score.append(image_label)
        labled_images_counter += 1
        print('finished ', round(labled_images_counter/len_test * 100, 2), '%' )
            
    # Calculate the accuracy
    accuracy = np.mean(images_score == test_labels)
    return accuracy

    
if __name__ == '__main__':
    train_images_path = 'train-images.idx3-ubyte'
    train_labels_path = 'train-labels.idx1-ubyte'
    test_images_path = 't10k-images.idx3-ubyte'
    test_labels_path = 't10k-labels.idx1-ubyte'

    train_images, train_labels, test_images, test_labels = create_dataset(train_images_path, train_labels_path, test_images_path, test_labels_path)
    examples_accuracy = examples_categorize(train_images, train_labels, test_images, test_labels)

    print("final accuracy of examples categorization: ", examples_accuracy * 100, '%')

    
    
    




