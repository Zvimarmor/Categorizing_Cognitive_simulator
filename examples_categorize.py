import numpy as np
import struct
import matplotlib.pyplot as plt

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

# Path to the uploaded file
file_path_images = 'train-images.idx3-ubyte'
file_path_labels = 'train-labels.idx1-ubyte'

# Read the image data
train_images = read_idx(file_path_images)
train_labels = read_idx(file_path_labels)

#leave only images with label 0 or 1
train_images = train_images[train_labels <= 1]
train_labels = train_labels[train_labels <= 1]


#path to the test file
file_path_images = 't10k-images.idx3-ubyte'
file_path_labels = 't10k-labels.idx1-ubyte'

# Read the image data
test_images = read_idx(file_path_images)
test_labels = read_idx(file_path_labels)

#leave only images with label 0 or 1
test_images = test_images[test_labels <= 1]
test_labels = test_labels[test_labels <= 1]


# Calculate the distance between the prototypes and the test images
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
print(f'Accuracy: {accuracy * 100:.2f}%', 'Size of test_labels:', len(test_labels), 'Size of train labels:', len(train_labels) )


