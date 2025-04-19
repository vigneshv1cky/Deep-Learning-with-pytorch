##################################################
#                   Part 1
# Chatper 2
##################################################

import torch

torch.version.__version__

a = torch.ones(3, 3)
b = torch.ones(3, 3)

a + b

##################################################
#       Pretrained Neural networks (Resnet)
##################################################

import torch
from torchvision import models, transforms
from PIL import Image
import torch

# Load pre-trained ResNet101 model
resnet = models.resnet101(pretrained=True)
resnet.eval()

# Define image preprocessing pipeline
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load and preprocess the image
img_path = "./data/p1ch2/bobby.jpg"
img = Image.open(img_path)
img

img_t = preprocess(img)
img_t

batch_t = torch.unsqueeze(img_t, 0)  # Add batch dimension
print(batch_t.shape)

# Perform inference
out = resnet(batch_t)
out

# Load ImageNet class labels
with open("./data/p1ch2/imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

len(labels)
labels[10]

# Get the top predicted class
_, index = torch.max(out, 1)
out[0, 207]

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

# Display top-1 prediction
predicted_label = labels[index[0]]
predicted_confidence = percentage[index[0]].item()
print(f"Top Prediction: {predicted_label} ({predicted_confidence:.2f}%)")

# Get top-5 predictions
_, indices = torch.sort(out, descending=True)
indices

top5_predictions = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]

# Display top-5 predictions
print("Top 5 Predictions:")
for label, confidence in top5_predictions:
    print(f"{label}: {confidence:.2f}%")


##################################################
#                   Cycle GAN
##################################################

import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, dim):
        """
        A single Residual Block used in the generator.

        Args:
        - dim (int): The number of input and output channels.
        """
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        """
        Builds the sequential convolutional block inside the residual block.

        Structure:
        - Reflection padding (1 pixel)
        - Conv2D (3x3, no padding because of reflection padding)
        - Instance Normalization
        - ReLU activation
        - Reflection padding again
        - Another Conv2D (3x3, same channels)
        - Instance Normalization
        """
        conv_block = []

        # Step 1: Reflection padding
        conv_block += [nn.ReflectionPad2d(1)]
        # Adds a 1-pixel border around the input to keep spatial dimensions unchanged
        # Output shape: (B, dim, H+2, W+2)

        # Step 2: First Convolution Layer
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            # 3x3 convolution with no padding (padding=0) since we already applied reflection padding
            # Output shape: (B, dim, H, W)
            nn.InstanceNorm2d(dim),  # Normalization per channel
            nn.ReLU(True),  # ReLU activation function: f(x) = max(0, x)
        ]

        # Step 3: Reflection Padding Again
        conv_block += [nn.ReflectionPad2d(1)]
        # Output shape: (B, dim, H+2, W+2)

        # Step 4: Second Convolution Layer
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            # Another 3x3 convolution, keeping the number of channels unchanged
            # Output shape: (B, dim, H, W)
            nn.InstanceNorm2d(dim),  # Normalize output again
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """
        Forward pass through the residual block.

        Mathematically:
        - Let x be the input tensor.
        - F(x) is the output of the convolutional layers.
        - The residual connection computes: y = x + F(x)

        This skip connection helps retain input features.
        """
        out = x + self.conv_block(
            x
        )  # Element-wise addition of input and transformed output
        return out


class ResNetGenerator(nn.Module):
    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        base_filters=64,
        num_residual_blocks=9,
    ):
        """
        A ResNet-based generator network for image-to-image translation.

        Args:
        - input_channels (int): Number of input channels (default: 3 for RGB)
        - output_channels (int): Number of output channels (default: 3 for RGB)
        - base_filters (int): Number of filters in the first convolutional layer.
        - num_residual_blocks (int): Number of residual blocks in the middle of the network.
        """
        assert num_residual_blocks >= 0
        super(ResNetGenerator, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_filters = base_filters

        # Step 1: Initial Convolution Block
        layers = [
            nn.ReflectionPad2d(
                3
            ),  # Adds 3 pixels of padding around the image to preserve spatial size
            nn.Conv2d(
                input_channels, base_filters, kernel_size=7, padding=0, bias=True
            ),
            # 7x7 convolution to capture large context information
            # Output shape: (B, 64, H, W)
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(True),
        ]

        # Step 2: Downsampling Layers (Encoder)
        num_downsampling_layers = 2
        for i in range(num_downsampling_layers):
            filter_multiplier = 2**i
            layers += [
                nn.Conv2d(
                    base_filters * filter_multiplier,
                    base_filters * filter_multiplier * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                ),
                # Stride=2 halves spatial resolution (H → H/2, W → W/2)
                nn.InstanceNorm2d(base_filters * filter_multiplier * 2),
                nn.ReLU(True),
            ]

        # Step 3: Bottleneck (Residual Blocks)
        bottleneck_filters = 2**num_downsampling_layers * base_filters
        for _ in range(num_residual_blocks):
            layers += [ResNetBlock(bottleneck_filters)]
            # Each residual block maintains spatial resolution

        # Step 4: Upsampling Layers (Decoder)
        for i in range(num_downsampling_layers):
            filter_multiplier = 2 ** (num_downsampling_layers - i)
            layers += [
                nn.ConvTranspose2d(
                    base_filters * filter_multiplier,
                    int(base_filters * filter_multiplier / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=True,
                ),
                # Stride=2 doubles spatial resolution (H/2 → H, W/2 → W)
                nn.InstanceNorm2d(int(base_filters * filter_multiplier / 2)),
                nn.ReLU(True),
            ]

        # Step 5: Final Convolutional Layer
        layers += [nn.ReflectionPad2d(3)]  # Padding before final conv
        layers += [nn.Conv2d(base_filters, output_channels, kernel_size=7, padding=0)]
        # Maps back to RGB channels
        layers += [nn.Tanh()]  # Tanh activation normalizes pixel values to [-1, 1]

        self.model = nn.Sequential(*layers)

    def forward(self, input_tensor):
        """
        Forward pass through the generator.

        The input is processed through:
        - Encoding (Convolution + Downsampling)
        - Transformation (Residual Blocks)
        - Decoding (Transposed Convolution + Upsampling)

        Mathematically:
        - F_enc(x): Encoder
        - F_res(F_enc(x)): Bottleneck residual blocks
        - F_dec(F_res(F_enc(x))): Decoder
        """
        return self.model(input_tensor)


netG = ResNetGenerator()
model_path = "./data/p1ch2/horse2zebra_0.4.0.pth"
model_data = torch.load(model_path)
netG.load_state_dict(model_data)
netG.eval()


from PIL import Image
from torchvision import transforms

preprocess = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
img = Image.open("./data/p1ch2/horse.jpg")
img

img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)
batch_out = netG(batch_t)
out_t = (batch_out.data.squeeze() + 1.0) / 2.0
out_img = transforms.ToPILImage()(out_t)
# out_img.save('./data/p1ch2/zebra.jpg')
out_img
