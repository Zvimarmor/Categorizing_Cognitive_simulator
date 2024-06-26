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

def show_prototype(prototype, num):
    '''
    This function displays the prototype image.
    '''
    plt.imshow(prototype, cmap='gray')
    plt.title('prototype ' +str(num) + 'from '+ str(len(train_images[train_labels == 0]))+ ' images')
    plt.show()
    plt.close()


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
    prototype_2 = np.mean(train_images[train_labels == 2], axis=0)
    prototype_3 = np.mean(train_images[train_labels == 3], axis=0)
    prototype_4 = np.mean(train_images[train_labels == 4], axis=0)
    prototype_5 = np.mean(train_images[train_labels == 5], axis=0)
    prototype_6 = np.mean(train_images[train_labels == 6], axis=0)
    prototype_7 = np.mean(train_images[train_labels == 7], axis=0)
    prototype_8 = np.mean(train_images[train_labels == 8], axis=0)
    prototype_9 = np.mean(train_images[train_labels == 9], axis=0)
    prototipes = [prototype_0, prototype_1, prototype_2, prototype_3, prototype_4, prototype_5, prototype_6, prototype_7, prototype_8, prototype_9]
    
    for i in prototipes:
        show_prototype(i, prototipes.index(i))

    # Calculate the distance between the prototypes and the test images
    images_score = []
    for test_image in test_images:
        possible_labels = []
        for i in prototipes:
            distance = np.mean((test_image - i) ** 2)
            possible_labels.append(distance)
        image_label = possible_labels.index(min(possible_labels))
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

    
    
    




