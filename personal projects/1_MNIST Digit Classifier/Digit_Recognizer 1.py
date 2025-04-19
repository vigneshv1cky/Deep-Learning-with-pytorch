######################################################################
# Step 1: Set Up the Environment
######################################################################
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

######################################################################
# Step 2: Load the MNIST Dataset
######################################################################
# The MNIST dataset is a collection of 28x28 grayscale images of handwritten digits (0-9).
(X_train, y_train), (X_test, y_test) = mnist.load_data()

######################################################################
# Step 3: Preprocess the Data
######################################################################
# Normalize the pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Print the shape of the datasets
print("Training data shape:", X_train.shape)  # (60000, 28, 28)
print("Testing data shape:", X_test.shape)  # (10000, 28, 28)
print("Training labels shape:", y_train.shape)  # (60000,)
print("Testing labels shape:", y_test.shape)  # (10000,)

# Visualize some sample images from the dataset
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")
plt.show()

######################################################################
# Step 4: Build a Neural Network Model
######################################################################
# Define the model
model = Sequential(
    [
        Flatten(
            input_shape=(28, 28)
        ),  # Flatten the 28x28 images into a 784-dimensional vector
        Dense(
            128, activation="relu"
        ),  # Fully connected layer with 128 units and ReLU activation
        Dense(
            10, activation="softmax"
        ),  # Output layer with 10 units (one for each digit) and softmax activation
    ]
)

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Print the model summary
model.summary()

######################################################################
# Step 5: Train the Model
######################################################################
# Train the model and store the history
history = model.fit(X_train, y_train, epochs=5, validation_split=0.2)

######################################################################
# Step 6: Evaluate the Model
######################################################################
# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

######################################################################
# Step 7: Plot Training History
######################################################################
# Plot the training and validation accuracy
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()

######################################################################
# Step 8: Make Predictions
######################################################################
# Make predictions on the test data
predictions = model.predict(X_test)

# Convert predictions from probabilities to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Visualize some test images with their predicted labels
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_test[i], cmap="gray")
    plt.title(f"Pred: {predicted_labels[i]}, True: {y_test[i]}")
    plt.axis("off")
plt.show()
