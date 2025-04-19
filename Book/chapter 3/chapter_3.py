# Import necessary libraries
import torch

# ==========================
# Basic List Operations
# ==========================
a = [1.0, 2.0, 1.0]  # Creating a list
print(a[0])  # Accessing the first element

a[2] = 3.0  # Modifying the list
print(a)

# ==========================
# Creating and Manipulating Tensors
# ==========================
a = torch.ones(3)  # Creating a tensor of ones
print(a)

# Accessing elements of a tensor
print(a[1])
print(float(a[1]))  # Converting to Python float

a[2] = 2.0  # Modifying tensor elements
print(a)

# ==========================
# Creating a Tensor from Scratch
# ==========================
points = torch.zeros(6)  # Initializing with zeros
points[0] = 4.0
points[1] = 1.0
points[2] = 5.0
points[3] = 3.0
points[4] = 2.0
points[5] = 1.0

# Creating a tensor from a list
points = torch.tensor([4.0, 1.0, 5.0, 3.0, 2.0, 1.0])
print(points)

# Accessing elements
print(float(points[0]), float(points[1]))

# ==========================
# Multi-Dimensional Tensors
# ==========================
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
print(points)

# Checking shape
print(points.shape)

# Creating zero tensors of specific shape
points = torch.zeros(3, 2)
print(points)

# Accessing specific elements
print(points[0, 1])  # Element at row 0, column 1
print(points[0])  # First row

# ==========================
# Tensor Storage and Memory
# ==========================
# Tensors in PyTorch have a storage object that holds the actual data.
# The storage is a contiguous block of memory where tensor elements are stored sequentially.
# This allows efficient memory access and operations.


print(points.storage())  # Inspecting storage

# Accessing storage directly
points_storage = points.storage()
print(points_storage[0])  # First element in storage
print(points.storage()[1])  # Second element

# Modifying tensor storage
points_storage[0] = 2.0
print(points)

# Storage offset, size, and stride
print(points[1].storage_offset())
print(points[1].size())
print(points.stride())

# ==========================
# Tensor Transposition
# ==========================

# Transposition swaps dimensions in a tensor, effectively flipping rows and columns.
# This does not create a new tensor but instead creates a new view of the same data with updated strides.

points_t = points.t()  # Transpose operation
print(points_t)
print(points_t.stride())  # Checking stride after transposing

# Checking if storage is shared
print(id(points.storage()) == id(points_t.storage()))

# ==========================
# Contiguous Memory and Transpose
# ==========================

# The `contiguous()` method creates a new tensor with the same data but stored in a contiguous block of memory.
# This is necessary when working with operations that require memory to be contiguous, like reshaping or some CUDA operations.

points_t_cont = points_t.contiguous()
print(points_t_cont)
print(points_t_cont.stride())


# ==========================
# Tensor Data Types
# ==========================
# Creating tensors with different data types
# `dtype=torch.double` specifies 64-bit floating point values
# `dtype=torch.short` specifies 16-bit integer values
double_points = torch.ones(10, 2, dtype=torch.double)
short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)
print(short_points.dtype)  # Checking data type

# Changing tensor data types
# Using `.to()` method to convert to a different type
short_points = short_points.to(torch.int16)
print(short_points.dtype)  # Checking updated data type

# ==========================
# Tensor Slicing
# ==========================
# Slicing works similarly to Python lists
some_list = list(range(6))
print(some_list[1:4])  # Extracting a slice (index 1 to 3)

# Slicing tensors
print(points[1:])  # Extracting all rows from index 1 onwards
print(points[:, 0])  # Extracting the first column from all rows

# Adding an extra dimension using None
print(points[None])  # Adds an extra dimension at axis 0

# ==========================
# Conversion between PyTorch and NumPy
# ==========================
# Converting a PyTorch tensor to a NumPy array
points_np = points.numpy()
print(points_np)

# Converting back from NumPy array to PyTorch tensor
points = torch.from_numpy(points_np)

# ==========================
# Saving and Loading Tensors
# ==========================
# Saving tensors to a file
torch.save(points, "ourpoints.t")

# Loading the tensor from file
points = torch.load("ourpoints.t")
print(points)

# ==========================
# Using HDF5 Format
# ==========================
# HDF5 (Hierarchical Data Format version 5) is a file format designed to store large amounts of numerical data.
# It supports datasets (similar to NumPy arrays) and groups (similar to directories) within a single file.
# HDF5 is useful for managing and organizing complex datasets efficiently, especially in scientific computing.

import h5py

# Saving tensor data in HDF5 format
with h5py.File("ourpoints.hdf5", "w") as f:
    dset = f.create_dataset("coords", data=points.numpy())

# Loading data from an HDF5 file
with h5py.File("ourpoints.hdf5", "r") as f:
    dset = f["coords"]
    last_points = torch.from_numpy(dset[-2:])  # Extracting last two rows
print(last_points)

# ==========================
# Using GPU with PyTorch
# ==========================
# Moving tensor to GPU if available
if torch.cuda.is_available():
    points_gpu = points.to(device="cuda")
    print(points_gpu)  # Tensor on GPU
    points_cpu = points_gpu.to(device="cpu")
    print(points_cpu)  # Tensor moved back to CPU

# ==========================
# Transposing and Stride
# ==========================
# Creating a tensor and transposing its dimensions
a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)  # Swapping rows and columns
print(a.shape, a_t.shape)  # Checking shapes before and after transposition

# Resetting all values in the tensor to zero
a.zero_()
print(a)


# ==============================================================================
# Named Tensors
# ==============================================================================

import torch

# ==========================
# Section 1: Tensor Initialization
# ==========================
# Define weights for RGB to grayscale conversion
weights = torch.tensor([0.2126, 0.7152, 0.0722])

# Create a random image tensor with shape [channels, rows, columns]
img_t = torch.randn(3, 5, 5)

# Create a batch of images with shape [batch, channels, rows, columns]
batch_t = torch.randn(2, 3, 5, 5)

# ==========================
# Section 2: Naive Grayscale Conversion (Mean over Channels)
# ==========================
img_gray_naive = img_t.mean(dim=-3)
batch_gray_naive = batch_t.mean(dim=-3)

# Print the shapes to verify
print("Naive grayscale shapes:", img_gray_naive.shape, batch_gray_naive.shape)

# ==========================
# Section 3: Weighted Grayscale Conversion (Manual Broadcasting)
# ==========================
# Reshape weights to align with image tensor
unsqueezed_weights = weights.unsqueeze(-1).unsqueeze_(-1)
unsqueezed_weights.shape

# Apply weighted sum across channels
img_weights = img_t * unsqueezed_weights
batch_weights = batch_t * unsqueezed_weights

# Summing over channels to get grayscale output
img_gray_weighted = img_weights.sum(dim=-3)
batch_gray_weighted = batch_weights.sum(dim=-3)

# Print shapes to verify
print(
    "Weighted grayscale shapes:",
    batch_weights.shape,
    batch_t.shape,
    unsqueezed_weights.shape,
)

# ==========================
# Section 4: Using Einstein Summation Notation
# ==========================
img_gray_weighted_fancy = torch.einsum("...chw,c->...hw", img_t, weights)
batch_gray_weighted_fancy = torch.einsum("...chw,c->...hw", batch_t, weights)

# Print the shape of the batch grayscale result
print("Einstein summation result shape:", batch_gray_weighted_fancy.shape)

# ==========================
# Section 5: Named Tensors (Experimental Feature)
# ==========================
# Creating named tensor for weights
weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=["channels"])

# Refining names for image and batch tensors
img_named = img_t.refine_names(..., "channels", "rows", "columns")
batch_named = batch_t.refine_names(..., "channels", "rows", "columns")

# Print named tensor details
print("img named:", img_named.shape, img_named.names)
print("batch named:", batch_named.shape, batch_named.names)

# Align weights to match image tensor's named dimensions
weights_aligned = weights_named.align_as(img_named)

# Print aligned weights shape and names
print("Aligned weights:", weights_aligned.shape, weights_aligned.names)

# Perform grayscale conversion using named tensors
gray_named = (img_named * weights_aligned).sum("channels")

# Print the shape of grayscale output
gray_named.shape, gray_named.names

# ==========================
# Section 6: Handling Naming Errors
# ==========================
try:
    gray_named = (img_named[..., :3] * weights_named).sum("channels")
except Exception as e:
    print("Error:", e)

# ==========================
# Section 7: Removing Names from Named Tensor
# ==========================
gray_plain = gray_named.rename(None)
print("Unnamed grayscale shape:", gray_plain.shape, gray_plain.names)
