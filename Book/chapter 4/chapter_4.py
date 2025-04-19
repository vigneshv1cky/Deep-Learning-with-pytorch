import numpy as np
import torch

torch.set_printoptions(edgeitems=2, threshold=50)

import imageio

img_arr = imageio.imread("../data/p1ch4/image-dog/bobby.jpg")
img_arr.shape


img = torch.from_numpy(img_arr)
out = img.permute(2, 0, 1)
# permute(2, 0, 1) rearranges the dimensions from [H, W, C] to [C, H, W],
# aligning the data format with what PyTorch typically expects for image inputs.


batch_size = 3
batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)


import os

data_dir = "../data/p1ch4/image-cats/"
filenames = [
    name for name in os.listdir(data_dir) if os.path.splitext(name)[-1] == ".png"
]
for i, filename in enumerate(filenames):
    img_arr = imageio.imread(os.path.join(data_dir, filename))
    img_t = torch.from_numpy(img_arr)
    img_t = img_t.permute(2, 0, 1)
    img_t = img_t[:3]  # <1>
    batch[i] = img_t

batch = batch.float()
batch /= 255.0

n_channels = batch.shape[1]
for c in range(n_channels):
    mean = torch.mean(batch[:, c])
    std = torch.std(batch[:, c])
    batch[:, c] = (batch[:, c] - mean) / std


# ========================================================
# 4.2 - 3D images: Volumetric data
# ========================================================


# ========================================================
# 4.3 Representing tabular data
# ========================================================

import csv

wine_path = "../data/p1ch4/tabular-wine/winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)
wineq_numpy

col_list = next(csv.reader(open(wine_path), delimiter=";"))
wineq_numpy.shape, col_list

wineq = torch.from_numpy(wineq_numpy)
wineq.shape, wineq.dtype

data = wineq[:, :-1]
data, data.shape

target = wineq[:, -1]
target, target.shape

target = wineq[:, -1].long()
target

target_onehot = torch.zeros(target.shape[0], 10)
target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

target_unsqueezed = target.unsqueeze(1)
target_unsqueezed

data_mean = torch.mean(data, dim=0)
data_mean

data_var = torch.var(data, dim=0)
data_var

bad_indexes = target <= 3
bad_indexes.shape, bad_indexes.dtype, bad_indexes.sum()

bad_data = data[bad_indexes]
bad_data.shape

bad_data = data[target <= 3]
mid_data = data[(target > 3) & (target < 7)]
good_data = data[target >= 7]

bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)
for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print("{:2} {:20} {:6.2f} {:6.2f} {:6.2f}".format(i, *args))


total_sulfur_threshold = 141.83
total_sulfur_data = data[:, 6]
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)
predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum()

actual_indexes = target > 5
actual_indexes.shape, actual_indexes.dtype, actual_indexes.sum()


n_matches = torch.sum(actual_indexes & predicted_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()
n_matches, n_matches / n_predicted, n_matches / n_actual
