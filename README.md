# Image Categorization Experiment

This project explores two methods of image categorization: prototype-based and examples-based categorization. The dataset used is a subset of the MNIST dataset containing only images labeled as 0 or 1. The project evaluates the accuracy of both methods by varying the percentage of training data used.

## Project Structure

- `main.py`: The main script containing the functions and the experiment loop.
- `train-images.idx3-ubyte`: The file containing training images.
- `train-labels.idx1-ubyte`: The file containing labels for the training images.
- `t10k-images.idx3-ubyte`: The file containing test images.
- `t10k-labels.idx1-ubyte`: The file containing labels for the test images.
- `Accuracy of prototype and examples-based categorization.png`: The output plot showing the accuracy of both methods.

## Dependencies

To run the project, you need the following Python libraries:
- `numpy`
- `matplotlib`

You can install them using pip:
```bash
pip install numpy matplotlib
```

## Usage

1. **Download and place the dataset files**: Ensure you have the following files in the same directory as the `main.py` script:
    - `train-images.idx3-ubyte`
    - `train-labels.idx1-ubyte`
    - `t10k-images.idx3-ubyte`
    - `t10k-labels.idx1-ubyte`

2. **Run the script**: Execute the script by running:
    ```bash
    python main.py
    ```

3. **View the results**: The script will generate a plot (`Accuracy of prototype and examples-based categorization.png`) showing the accuracy of prototype-based and examples-based categorization methods as a function of the percentage of training data used.

## Functions

### `read_idx(filename)`
Reads an IDX file and returns the data as a NumPy array.

### `create_dataset(train_images_path, train_labels, test_images, test_labels)`
Reads the training and testing data from the specified files and returns the data, only including images with labels 0 or 1.

### `create_part_of_all_dataset(train_images_path, train_labels, test_images, test_labels, percentage)`
Reads the training and testing data, and returns a subset of the training data based on the specified percentage, only including images with labels 0 or 1.

### `prototype_categorized(train_images, train_labels, test_images, test_labels)`
Calculates the accuracy of prototype-based categorization by creating prototypes for each class and assigning the label of the closest prototype to the test images.

### `examples_categorize(train_images, train_labels, test_images, test_labels)`
Calculates the accuracy of examples-based categorization by calculating the distance between each test image and each training image, and assigning the label of the closest training image to the test images.

## Notes

- The `examples_categorize` function can take a long time to run due to the nested loops calculating distances between each test image and every training image.
- The script evaluates the accuracy of both methods by varying the percentage of training data used from 1% to 100%.

## Results

The output plot shows the accuracy of both categorization methods. The x-axis represents the percentage of training data used, and the y-axis represents the accuracy.
