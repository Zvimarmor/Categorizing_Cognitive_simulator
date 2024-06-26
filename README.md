# Image Categorization Experiment

This project explores two methods of image categorization: prototype-based and examples-based categorization. The dataset used is a subset of the MNIST dataset containing only images labeled as 0 or 1. The project evaluates the accuracy of both methods by varying the percentage of training data used.

## Project Structure

- **main.py**: The main script containing the functions and the experiment loop.
- **train-images.idx3-ubyte**: The file containing training images.
- **train-labels.idx1-ubyte**: The file containing labels for the training images.
- **t10k-images.idx3-ubyte**: The file containing test images.
- **t10k-labels.idx1-ubyte**: The file containing labels for the test images.
- **Accuracy of prototype and examples-based categorization.png**: The output plot showing the accuracy of both methods.

### Additional Files
- **all_digits_prototype_categorize.py**: This script extends the prototype-based categorization to include all digit labels (0-9). It generates prototypes for each digit and calculates the categorization accuracy.
- **all_digits_examples_categorize.py**: This script extends the examples-based categorization to include all digit labels (0-9). It calculates the accuracy of categorization by comparing each test image to all training images.

## Dependencies

To run the project, you need the following Python libraries:

- numpy
- matplotlib

## Usage

### Download and Place the Dataset Files
Ensure you have the following files in the same directory as the `main.py` script:

- train-images.idx3-ubyte
- train-labels.idx1-ubyte
- t10k-images.idx3-ubyte
- t10k-labels.idx1-ubyte

### Run the Main Script
Execute the main script

### Run the Additional Scripts
To run the scripts that include all digit labels (0-9):

#### all_digits_prototype_categorize.py
This script calculates the accuracy of prototype-based categorization for all digit labels (0-9) and displays the prototypes.

```bash
python all_digits_prototype_categorize.py
```

#### all_digits_examples_categorize.py
This script calculates the accuracy of examples-based categorization for all digit labels (0-9). Note that this script may take a long time to run due to the high computational complexity.

```bash
python all_digits_examples_categorize.py
```

### View the Results
The scripts will generate plots showing the accuracy of the categorization methods. The main script will save the plot as `Accuracy of prototype and examples-based categorization.png`.

## Functions

### read_idx(filename)
Reads an IDX file and returns the data as a NumPy array.

### create_dataset(train_images_path, train_labels, test_images, test_labels)
Reads the training and testing data from the specified files and returns the data, only including images with labels 0 or 1.

### create_part_of_all_dataset(train_images_path, train_labels, test_images, test_labels, percentage)
Reads the training and testing data, and returns a subset of the training data based on the specified percentage, only including images with labels 0 or 1.

### prototype_categorized(train_images, train_labels, test_images, test_labels)
Calculates the accuracy of prototype-based categorization by creating prototypes for each class and assigning the label of the closest prototype to the test images.

### examples_categorize(train_images, train_labels, test_images, test_labels)
Calculates the accuracy of examples-based categorization by calculating the distance between each test image and each training image, and assigning the label of the closest training image to the test images.

## Notes

- The `examples_categorize` function can take a long time to run due to the nested loops calculating distances between each test image and every training image.
- The script evaluates the accuracy of both methods by varying the percentage of training data used from 1% to 100%.

## Results

The output plot shows the accuracy of both categorization methods. The x-axis represents the percentage of training data used, and the y-axis represents the accuracy.

The additional scripts provide a more comprehensive analysis by including all digit labels (0-9), giving a broader perspective on the performance of the categorization methods.