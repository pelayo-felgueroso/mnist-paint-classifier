# MNIST Paint Digit Classifier

A simple convolutional neural network (CNN) built with TensorFlow/Keras to classify handwritten digits using the MNIST dataset. It also includes a tool to recognize digits drawn in external software (like Paint) on a black background with a white number.

This project is designed as a beginner-friendly yet slightly more advanced introduction to convolutional neural networks. It goes beyond the typical minimal examples by including:

- Batch normalization  
- ELU activations  
- Dropout regularization  

More importantly, it focuses on real usability: instead of limiting recognition to internal MNIST test images, it enables direct classification of digits drawn externally, making it practical, visual, and more engaging.

## Who is this for?

- Students starting with AI or deep learning who want to test models with their own images.  
- Teachers looking for a quick, visual demo of digit recognition.  
- Anyone who wants to train a CNN and classify real-world inputs without setting up complex pipelines or external datasets.  

## Project Structure

Mnist_numbers/ ├── mnist_cnn.py # Script to train the CNN model using MNIST ├── test_model.py # Script to test the model with custom PNG images ├── numberdetector.h5 # Trained model saved in HDF5 format └── numerospaint/ # Folder containing hand-drawn digit images (PNG format) ├── zero.png ├── one.png ├── ... └── nine.png

markdown
Copiar
Editar

## Features

- Trains a CNN on the MNIST dataset with advanced layers.
- Saves the model as an `.h5` file.
- Allows prediction from hand-drawn images (square, grayscale).

## Results

- Achieves **~99% accuracy** on MNIST test set.  
- Achieves **100% accuracy** when classifying Paint images with clear digit shapes and correct format.  

## Requirements

- Python 3.7+  
- TensorFlow  
- NumPy  
- OpenCV  
- Matplotlib  

If you need help setting up TensorFlow with GPU support, you can follow this guide:  
[TensorFlow GPU Setup by pelayo-felgueroso](https://github.com/pelayo-felgueroso/tensorflow-gpu-setup)

## How to Use

### 1. Train the Model

Run `mnist_cnn.py` to train the CNN and save the model:

```bash
python mnist_cnn.py
This generates a file called numberdetector.h5.

2. Test with Custom Images
Place your hand-drawn digit PNGs (black background, white number, square aspect ratio) into the numerospaint/ folder. Then run:

bash
Copiar
Editar
python test_model.py
The script will:

Load the specified image

Preprocess and resize it to 28x28

Predict the digit and display the confidence

Example Output
csharp
Copiar
Editar
The model predicts the image is the number: 4, with confidence: 0.99987
Image Format Guidelines
Format: PNG

Aspect Ratio: Square

Background: Black

Digit: White (centered and clearly drawn)

Author
Pelayo Felgueroso

Feel free to fork, modify, and improve this repository. It's designed for learning and experimentation with computer vision and CNNs using TensorFlow.