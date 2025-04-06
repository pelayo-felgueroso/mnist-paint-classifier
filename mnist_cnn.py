import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the CNN model
model = models.Sequential()

# Convolutional layer 1
model.add(layers.Conv2D(32, (5, 5), activation=None, input_shape=(28, 28, 1), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.ELU())
model.add(layers.MaxPooling2D((2, 2)))

# Convolutional layer 2
model.add(layers.Conv2D(64, (5, 5), activation=None, padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.ELU())
model.add(layers.MaxPooling2D((2, 2)))

# Convolutional layer 3
model.add(layers.Conv2D(128, (5, 5), activation=None, padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.ELU())
model.add(layers.MaxPooling2D((2, 2)))

# Convolutional layer 4
model.add(layers.Conv2D(256, (5, 5), activation=None, padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.ELU())
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the convolutional output (1x1x256 -> 256)
model.add(layers.Flatten())

# Fully connected layer
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))  # Dropout to prevent overfitting (randomly disables 50% of the neurons)

# Output layer with Softmax activation (for 10-class classification)
model.add(layers.Dense(10, activation='softmax'))

# Compile the model using Adam optimizer and categorical crossentropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Save the trained model to an H5 file
model.save('numberdetector.h5')
