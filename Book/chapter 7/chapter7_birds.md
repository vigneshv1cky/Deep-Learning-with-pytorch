This explanation breaks down how to build a dataset from CIFAR-10 to distinguish between birds and airplanes, construct a fully connected neural network to classify them, convert the network’s output into probabilities, choose an appropriate loss function, train the model, and finally discuss some of its limitations.

---

## 1. Building the Dataset

Jane’s bird-watching club only cares about airplanes and birds. In CIFAR‑10, airplanes are labeled with 0 and birds with 2. We filter the dataset to include only these classes and then remap the labels so they’re contiguous (0 for airplane and 1 for bird):

```python
# Map original labels to new ones: 0 (airplane) -> 0, 2 (bird) -> 1
label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']

# Filter training data: include only images where label is 0 or 2
cifar2 = [(img, label_map[label]) for img, label in cifar10 if label in [0, 2]]

# Similarly, filter the validation dataset
cifar2_val = [(img, label_map[label]) for img, label in cifar10_val if label in [0, 2]]
```

*Comments:*  

- **Filtering:** We iterate over CIFAR‑10 and only keep the images that are either airplanes or birds.  
- **Remapping:** Since the original labels (0 and 2) are not contiguous (i.e., not 0 and 1), we remap them for convenience.

---

## 2. Building a Fully Connected Neural Network

Since the CIFAR‑10 images are 32×32 RGB images, each image has 32 × 32 × 3 = 3072 pixel values. A fully connected (or “dense”) network treats the image as a flat vector.

```python
import torch.nn as nn

# Number of output classes: 2 (airplane and bird)
n_out = 2  

# Define a simple fully connected network
model = nn.Sequential(
    nn.Linear(3072, 512),  # Input layer: 3072 features -> 512 hidden neurons
    nn.Tanh(),             # Activation function to introduce non-linearity
    nn.Linear(512, n_out)   # Output layer: 512 hidden neurons -> 2 class scores
)
```

*Comments:*  

- The first layer flattens and maps the 3072 input features to 512 hidden features.  
- The Tanh activation enables the network to learn non-linear relationships.  
- The final layer outputs two scores (one for each class).

---

## 3. Converting Outputs to Probabilities with Softmax

The raw outputs (or logits) need to be converted into probabilities that sum to 1. The softmax function does exactly that.

### Manual Softmax Function

```python
import torch

def softmax(x):
    # Exponentiate each element and then normalize by the sum of all exponentials
    return torch.exp(x) / torch.exp(x).sum()

# Test on a sample vector
x = torch.tensor([1.0, 2.0, 3.0])
print(softmax(x))
# Expected output: tensor([0.0900, 0.2447, 0.6652])
```

### Using PyTorch’s Built-In Softmax for Batches

```python
# For batch processing, specify the dimension along which softmax should be applied.
softmax_layer = nn.Softmax(dim=1)

# Example with a batch of two vectors (rows)
x_batch = torch.tensor([[1.0, 2.0, 3.0],
                        [1.0, 2.0, 3.0]])
print(softmax_layer(x_batch))
```

*Comments:*  

- Softmax exponentiates each element and normalizes so that the sum equals 1.  
- When using batches, setting `dim=1` tells PyTorch to apply softmax across the feature dimension (each row).

---

## 4. Choosing the Loss Function for Classification

For classification, we want our network to learn to assign a high probability to the correct class. Instead of using Mean Squared Error (MSE), we use a loss that penalizes misclassifications. In PyTorch, one common approach is to use negative log likelihood (NLL) loss. Since NLL expects log probabilities as inputs, we replace the softmax with a LogSoftmax layer.

### Modifying the Model

```python
model = nn.Sequential(
    nn.Linear(3072, 512),
    nn.Tanh(),
    nn.Linear(512, 2),
    nn.LogSoftmax(dim=1)  # Converts raw scores to log probabilities in a numerically stable way
)
```

### Defining the Loss and Testing on a Sample

```python
# Define the loss function for classification
loss_fn = nn.NLLLoss()

# Take one sample from the filtered dataset (e.g., a bird image)
img, label = cifar2[0]

# Reshape the image: flatten and add a batch dimension
img_batch = img.view(-1).unsqueeze(0)

# Forward pass: get model output (log probabilities)
output = model(img_batch)

# Compute the negative log likelihood loss
loss_value = loss_fn(output, torch.tensor([label]))
print(loss_value)
```

*Comments:*  

- The combination of `nn.LogSoftmax` and `nn.NLLLoss` allows us to penalize the model when it assigns a low probability to the correct class.  
- For binary classification, this is equivalent in purpose to using `nn.CrossEntropyLoss` (which combines the two steps).

---

## 5. Training the Classifier

### Training Loop Over Single Samples

Initially, the training loop can process one image at a time:

```python
import torch.optim as optim

learning_rate = 1e-2
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
n_epochs = 100

# Loop over epochs
for epoch in range(n_epochs):
    for img, label in cifar2:
        # Flatten image and add batch dimension
        output = model(img.view(-1).unsqueeze(0))
        loss_value = loss_fn(output, torch.tensor([label]))
        
        optimizer.zero_grad()  # Clear previous gradients
        loss_value.backward()  # Backpropagate to compute new gradients
        optimizer.step()       # Update the model's parameters

    print("Epoch: %d, Loss: %f" % (epoch, float(loss_value)))
```

*Comments:*  

- We iterate over each epoch and each training sample, computing the loss, backpropagating, and updating parameters with SGD.

### Training with Minibatches Using DataLoader

For better gradient estimation and stability, we use minibatches:

```python
from torch.utils.data import DataLoader

# Create a DataLoader for minibatch training (e.g., batch size = 64)
train_loader = DataLoader(cifar2, batch_size=64, shuffle=True)

for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        batch_size = imgs.shape[0]
        # Flatten all images in the batch
        outputs = model(imgs.view(batch_size, -1))
        loss_value = loss_fn(outputs, labels)
        
        optimizer.zero_grad()  # Reset gradients for the batch
        loss_value.backward()  # Backpropagation on the batch
        optimizer.step()       # Update model parameters

    print("Epoch: %d, Loss: %f" % (epoch, float(loss_value)))
```

*Comments:*  

- **DataLoader:** This utility shuffles data at every epoch and creates minibatches, which helps the training process converge more stably.  
- **Batch Processing:** Flattening the batch of images and computing the loss over the entire minibatch improves gradient estimation.

---

## 6. Evaluating the Model

After training, we evaluate our model on the validation set to measure accuracy:

```python
# Create a DataLoader for the validation set
val_loader = DataLoader(cifar2_val, batch_size=64, shuffle=False)
correct = 0
total = 0

# Disable gradient computation during evaluation
with torch.no_grad():
    for imgs, labels in val_loader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        # Get the index of the highest log-probability (predicted class)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())

accuracy = correct / total
print("Accuracy: %f" % accuracy)
```

*Comments:*  

- We compare the predicted class (via the argmax over the output probabilities) with the true labels and compute the overall accuracy.

---

## 7. Limitations of Fully Connected Layers for Image Tasks

The excerpt highlights a key shortcoming of fully connected networks when used with images:

- **Lack of Translation Invariance:**  
  Fully connected layers treat images as one long vector, meaning the spatial relationships between pixels (for example, parts of an airplane) are lost. As a result, a plane shifted by a few pixels might not be recognized correctly.

- **Excessive Parameters:**  
  The fully connected approach quickly becomes infeasible as image resolution increases. For example, a 1024×1024 image has over 3 million pixels, and a fully connected layer connecting this to even 1024 neurons would have billions of parameters—impractical for most hardware.

*Conclusion:*  
Because of these limitations, although our fully connected network can learn to distinguish between birds and airplanes on a small dataset like CIFAR‑10, it isn’t scalable or robust for real-world image recognition tasks. Convolutional Neural Networks (CNNs), which leverage local spatial structure and are translation invariant, are the preferred architecture for image data—and they will be covered in later chapters.

---
