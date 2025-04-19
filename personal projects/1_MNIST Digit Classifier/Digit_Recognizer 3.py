######################################################################
# Set Up the Environment
######################################################################
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import LearningRateScheduler, EarlyStopping, TensorBoard
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# Create a directory to save model and plots
os.makedirs("output", exist_ok=True)

######################################################################
# Load the MNIST Dataset
######################################################################
# The MNIST dataset is a collection of 28x28 grayscale images of handwritten digits (0-9).
(X_train, y_train), (X_test, y_test) = mnist.load_data()

######################################################################
# Preprocess the Data
######################################################################
# Normalize the pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Print the shape of the datasets
print("Training data shape:", X_train.shape)  # (60000, 28, 28)
print("Testing data shape:", X_test.shape)  # (10000, 28, 28)
print("Training labels shape:", y_train.shape)  # (60000,)
print("Testing labels shape:", y_test.shape)  # (10000,)

######################################################################
# Visualize Sample Data
######################################################################
# Visualize some sample images from the dataset
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")
plt.suptitle("Sample Training Images", fontsize=16)
plt.savefig("output/sample_training_images.png")  # Save the plot
plt.show()

######################################################################
# Build a Neural Network Model
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
        Dropout(0.2),  # Dropout layer to prevent overfitting
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
# Add Callbacks (Learning Rate Scheduling and Early Stopping)
######################################################################
# Learning Rate Scheduling
def lr_scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * 0.1  # Reduce learning rate after 2 epochs


lr_callback = LearningRateScheduler(lr_scheduler)

# Early Stopping
early_stopping = EarlyStopping(
    monitor="val_loss", patience=2, restore_best_weights=True
)

# TensorBoard Callback
tensorboard_callback = TensorBoard(log_dir="./logs")

######################################################################
# Train the Model
######################################################################
# Train the model and store the history
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_split=0.2,
    callbacks=[lr_callback, early_stopping, tensorboard_callback],
)

######################################################################
# Evaluate the Model
######################################################################
# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

######################################################################
# Plot Training History
######################################################################
# Plot the training and validation accuracy and loss
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

plt.suptitle("Training History", fontsize=16)
plt.savefig("output/training_history.png")  # Save the plot
plt.show()

######################################################################
# Make Predictions
######################################################################
# Make predictions on the test data
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

######################################################################
# Confusion Matrix
######################################################################
# Generate and plot the confusion matrix
conf_matrix = confusion_matrix(y_test, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10),
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("output/confusion_matrix.png")  # Save the plot
plt.show()

######################################################################
# Classification Report
######################################################################
# Print precision, recall, and F1-score for each class
print("Classification Report:")
print(classification_report(y_test, predicted_labels))


######################################################################
# Visualize Predictions (Interactive)
######################################################################
# Allow user to input an index to visualize a specific test image and its prediction
def visualize_prediction(index):
    plt.imshow(X_test[index], cmap="gray")
    plt.title(f"Pred: {predicted_labels[index]}, True: {y_test[index]}")
    plt.axis("off")
    plt.savefig(f"output/prediction_{index}.png")  # Save the plot
    plt.show()


# Interactive visualization
index = int(
    input("Enter an index (0-9999) to visualize a test image and its prediction: ")
)
if 0 <= index < 10000:
    visualize_prediction(index)
else:
    print("Invalid index. Please enter a number between 0 and 9999.")


######################################################################
# Misclassified Examples
######################################################################
# Analyze and visualize misclassified examples
misclassified_idx = np.where(predicted_labels != y_test)[0]
plt.figure(figsize=(10, 10))
for i, idx in enumerate(misclassified_idx[:25]):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_test[idx], cmap="gray")
    plt.title(f"Pred: {predicted_labels[idx]}, True: {y_test[idx]}")
    plt.axis("off")
plt.suptitle("Misclassified Examples", fontsize=16)
plt.savefig("output/misclassified_examples.png")  # Save the plot
plt.show()

######################################################################
# TensorBoard Integration
######################################################################
# TensorBoard logs are already being saved in the `logs` directory.
# To view them, run the following command in the terminal:
# tensorboard --logdir=./logs

######################################################################
# Export Predictions
######################################################################
# Export predictions to a CSV file
import pandas as pd

results = pd.DataFrame({"True Label": y_test, "Predicted Label": predicted_labels})
results.to_csv("output/predictions.csv", index=False)
print("Predictions exported to output/predictions.csv")


######################################################################
# Save and Load the Model
######################################################################
# Save the trained model to disk
model.save("output/mnist_model.h5")
print("Model saved as output/mnist_model.h5")

# Load the model back (for demonstration)
from keras.models import load_model

loaded_model = load_model("output/mnist_model.h5")
print("Model loaded successfully!")
