
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Select GPU if available
device = torch.device("cpu")

print("Using device:", device)

# Convert images to tensors
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load content image
content_image = Image.open("images/input.png").convert("RGB")
content = transform(content_image).unsqueeze(0).to(device)

# Load style image
style_image = Image.open("images/style.png").convert("RGB")
style = transform(style_image).unsqueeze(0).to(device)

print("Content shape:", content.shape)
print("Style shape:", style.shape)


from torchvision.models import vgg19, VGG19_Weights

# Load pretrained VGG19 model
weights = VGG19_Weights.DEFAULT
model = vgg19(weights=weights).features.to(device).eval()

# Freeze model parameters
for parameter in model.parameters():
    parameter.requires_grad_(False)

print("VGG19 model loaded successfully!")


# Layers used for style transfer
style_layers = ["0", "5", "10", "19", "28"]
content_layer = "21"


# ImageNet normalization values
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

# Normalize image before VGG19
def normalize(image):
    return (image - mean) / std


def extract_features(image, model):
    features = {}

    for name, layer in model._modules.items():
        image = layer(image)

        if name in style_layers or name == content_layer:
            features[name] = image

        if name == "28":
            break

    return features



# Extract features from both images
with torch.no_grad():
    content_features = extract_features(content, model)
    style_features = extract_features(style, model)

print("Content feature shape:", content_features[content_layer].shape)
print("Style layers extracted:", len(style_layers))


import torch.nn.functional as F

# Calculate content loss
def content_loss(generated, target):
    return F.mse_loss(generated, target)


# Calculate Gram Matrix
def gram_matrix(features):
    batch, channels, height, width = features.shape

    features = features.reshape(batch, channels, height * width)

    gram = torch.bmm(features, features.transpose(1, 2))

    return gram / (channels * height * width)


# Calculate style loss
def style_loss(generated_features, target_features):
    loss = 0

    for layer in style_layers:
        generated_gram = gram_matrix(generated_features[layer])
        target_gram = gram_matrix(target_features[layer])

        loss += F.mse_loss(generated_gram, target_gram)

    return loss

#Calculate total variation loss
def total_variation_loss(image):
    horizontal = torch.abs(image[:,:,:, 1:] - image[:, :, :, :-1])
    vertical = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])   

    return horizontal.mean() + vertical.mean() 


# Test loss functions
with torch.no_grad():
    test_content_loss = content_loss(
        content_features[content_layer],
        content_features[content_layer]
    )

    test_style_loss = style_loss(
        content_features,
        style_features
    )

print("Content loss:", test_content_loss.item())
print("Style loss:", test_style_loss.item())


from torchvision.utils import save_image

# Start with a copy of the content image
generated = content.clone().detach().requires_grad_(True)

# Create optimizer
optimizer = torch.optim.Adam([generated], lr=0.02)

# Training settings
steps = 200
content_weight = 10
style_weight = 100000
tv_weight = 2

print("Starting style transfer...")

for step in range(1, steps + 1):
    optimizer.zero_grad()

    # Extract generated image features
    generated_features = extract_features(generated, model)

    # Calculate losses
    c_loss = content_loss(
        generated_features[content_layer],
        content_features[content_layer]
    )

    s_loss = style_loss(generated_features, style_features)

  
# Calculate smoothness loss
tv_loss = total_variation_loss(generated)

# Combine all three losses
total_loss = content_weight * c_loss + style_weight * s_loss + tv_weight * tv_loss


    # Update generated image
total_loss.backward()
optimizer.step()

    # Keep pixel values between 0 and 1
with torch.no_grad():
        generated.clamp_(0, 1)

if step % 10 == 0:
        print(f"Step {step}/{steps} - Loss: {total_loss.item():.4f}")

# Save final result
save_image(generated.detach(), "images/stylized_tv2.png")

print("Style transfer completed!")
