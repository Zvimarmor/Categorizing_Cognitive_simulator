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

    #leave only images with label 0 or 1
    train_images = train_images[train_labels <= 1]
    train_labels = train_labels[train_labels <= 1]
    test_images = test_images[test_labels <= 1]
    test_labels = test_labels[test_labels <= 1]
    
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

    #leave only images with label 0 or 1
    train_images = train_images[train_labels <= 1]
    train_labels = train_labels[train_labels <= 1]
    test_images = test_images[test_labels <= 1]
    test_labels = test_labels[test_labels <= 1]

    #take only a percentage of the train data and labels randomly
    if percentage < 1:
        train_images = train_images[:int(len(train_images)*percentage)]
        train_labels = train_labels[:int(len(train_labels)*percentage)]

    return train_images, train_labels, test_images, test_labels

def prototype_categorized(train_images, train_labels, test_images, test_labels):
    '''
    This function calculates the accuracy of the prototype-based categorization
    of the test images.
    The function uses the train_images and train_labels to create prototypes
    for each class. The function then calculates the distance between each test
    image and the prototypes, and assigns the label of the closest prototype
    to the test image.
    The function returns the accuracy of the categorization.
    '''
    # Create a prototype for each class
    prototype_0 = np.mean(train_images[train_labels == 0], axis=0)
    prototype_1 = np.mean(train_images[train_labels == 1], axis=0)

    #show the prototypes 
    # plt.imshow(prototype_0, cmap='gray')
    # plt.title('prototype 0 from '+ str(len(train_images[train_labels == 0]))+ ' images')
    # plt.show()
    # plt.close()

    # plt.imshow(prototype_1, cmap='gray')
    # plt.title('prototype 1 from '+ str(len(train_images[train_labels == 0]))+ ' images')
    # plt.show()
    # plt.close()

    # Calculate the distance between the prototypes and the test images
    images_score = []
    for test_image in test_images:
        distance_0 = np.mean((test_image - prototype_0) ** 2)
        distance_1 = np.mean((test_image - prototype_1) ** 2)
        image_label = 0 if distance_0 < distance_1 else 1
        images_score.append(image_label)

    # Calculate the accuracy
    accuracy = np.mean(images_score == test_labels)
    return accuracy

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
    for test_image in test_images:
        best_distance = np.inf
        image_label = None
        for i, train_image in enumerate(train_images):
            distance = np.mean((test_image - train_image) ** 2)
            if distance < best_distance:
                best_distance = distance
                image_label = train_labels[i]     
        images_score.append(image_label)
            
        # Calculate the accuracy
    accuracy = np.mean(images_score == test_labels)
    return accuracy
    
if __name__ == '__main__':
    train_images_path = 'train-images.idx3-ubyte'
    train_labels_path = 'train-labels.idx1-ubyte'
    test_images_path = 't10k-images.idx3-ubyte'
    test_labels_path = 't10k-labels.idx1-ubyte'

    prototype_accuracies = []
    examples_accuracies = []
    for percentage in range(1, 101):
        train_images, train_labels, test_images, test_labels = create_part_of_all_dataset(train_images_path, train_labels_path, test_images_path, test_labels_path, percentage/100)
        prototype_accuracy = prototype_categorized(train_images, train_labels, test_images, test_labels)
        prototype_accuracies.append(prototype_accuracy)
        examples_accuracy = examples_categorize(train_images, train_labels, test_images, test_labels) #warning: takes a long time to run
        examples_accuracies.append(examples_accuracy)
        print('finished', percentage, '%')

    plt.plot(range(1, 101), prototype_accuracies, label='Prototype-based categorization')
    plt.plot(range(1, 101), examples_accuracies, label='Examples-based categorization') 
    plt.title('Accuracy of categorization of test images')
    plt.xlabel('Percentage of training data used')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('Accuracy of prototype and examples-based categorization.png')
    plt.show()
    plt.close()

    
    
    




