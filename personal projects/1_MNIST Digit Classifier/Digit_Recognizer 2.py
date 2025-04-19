# Step 1: Set Up the Environment
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Step 2: Load the MNIST Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Step 3: Preprocess the Data
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
plt.suptitle("Sample Training Images", fontsize=16)
plt.show()

# Step 4: Build a Neural Network Model
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


# Step 5: Add Learning Rate Scheduling
def lr_scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * 0.1  # Reduce learning rate after 2 epochs


lr_callback = LearningRateScheduler(lr_scheduler)

# Step 6: Add Early Stopping
early_stopping = EarlyStopping(
    monitor="val_loss", patience=2, restore_best_weights=True
)

# Step 7: Train the Model
# Train the model and store the history
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_split=0.2,
    callbacks=[lr_callback, early_stopping],
)

# Step 8: Evaluate the Model
# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Step 9: Plot Training History
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

plt.suptitle("Training History", fontsize=16)
plt.show()

# Step 10: Make Predictions
# Make predictions on the test data
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Step 11: Confusion Matrix
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
plt.show()

# Step 12: Classification Report
# Print precision, recall, and F1-score for each class
print("Classification Report:")
print(classification_report(y_test, predicted_labels))


# Step 13: Visualize Predictions
# Allow user to input an index to visualize a specific test image and its prediction
def visualize_prediction(index):
    plt.imshow(X_test[index], cmap="gray")
    plt.title(f"Pred: {predicted_labels[index]}, True: {y_test[index]}")
    plt.axis("off")
    plt.show()


# Interactive visualization
index = int(
    input("Enter an index (0-9999) to visualize a test image and its prediction: ")
)
if 0 <= index < 10000:
    visualize_prediction(index)
else:
    print("Invalid index. Please enter a number between 0 and 9999.")


# Step 14: Save and Load the Model
# Save the trained model to disk
model.save("mnist_model.h5")
print("Model saved as mnist_model.h5")

# Load the model back (for demonstration)
from keras.models import load_model

loaded_model = load_model("mnist_model.h5")
print("Model loaded successfully!")
