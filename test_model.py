import os
import numpy as np
import cv2  # OpenCV
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Step 1: Load the previously trained model
Deus = load_model('numberdetector.h5')  # Make sure the model path is correct
Deus.summary()

################## PAINT #######################
# Load the image you drew in Paint: White number in black background (square image)
img_path = os.path.join('.', 'numerospaint', 'seven.png')

# Load the image in grayscale
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Display the grayscale image for verification
plt.imshow(img, cmap='gray')
plt.title("Grayscale image from Paint")
plt.show()

################# MNIST #################################
# (Optional) Load an image from the MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# Select a random MNIST image
# random_index = np.random.randint(0, len(x_test))  # Generate a random index
# img = x_test[random_index]  # Select a random image from the test set
# true_label = y_test[random_index]  # True label of the selected image

# Display the MNIST grayscale image
# plt.imshow(img, cmap='gray')
# plt.title(f"Original MNIST image: {true_label}")
# plt.show()

########################################################
# Resize the image to 28x28 pixels (same size as MNIST images)
img_resized = cv2.resize(img, (28, 28))

# Normalize pixel values to [0, 1]
img_resized = img_resized.astype('float32') / 255.0

# Add the channel dimension (grayscale), as the model expects shape (28, 28, 1)
img_resized = np.expand_dims(img_resized, axis=-1)  # Shape becomes (28, 28, 1)

# Add the batch dimension (the model expects a batch of images)
img_resized = np.expand_dims(img_resized, axis=0)  # Shape becomes (1, 28, 28, 1)

# (Optional) Check preprocessed image dimensions
# print(f'Preprocessed image shape: {img_resized.shape}')  # Should be (1, 28, 28, 1)

# (Optional) Check image data type
# print(f"Image data type: {img_resized.dtype}")  # Should be float32

# Predict the digit with the loaded model
predictions = Deus.predict(img_resized)

# Get the predicted class (digit) with the highest probability
predicted_class = np.argmax(predictions)
predicted_confidence = np.max(predictions)

# Display the result
print(f"The model predicts the image is the number: {predicted_class}, with confidence: {predicted_confidence}")
