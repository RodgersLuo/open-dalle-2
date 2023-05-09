import torch
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from diffusion import sample_timestep, Diffusion
from unet import UNet

import sys
sys.path.insert(0, 'dataset')
sys.path.insert(0, 'nn_components')
from dataset import ImageCaptionDataset, load_data
from tokenizer import tokenize

# Define hyperparameters
T = 300
BATCH_SIZE = 256
IMG_SIZE = 64
EPOCHS = 1000
LR = 0.001
GRAD_CLIP = 0.008
NULL_EMB_RATE = 0.2
GUIDANCE_SCALE = 1

# UNet transformer
N_VOCAB=49408
CONTEXT_LENGTH=77
TRANSFORMER_WIDTH=64
TRANSFORMER_LAYERS=4
TRANSFORMER_HEADS=8
QKV_HEADS = 8

# The null caption embedding
NULL_TOKEN = torch.zeros(CONTEXT_LENGTH, dtype=torch.int)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


def train(model, dataloader, diffusion):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        for step, (img, txt) in enumerate(dataloader):
            # model.train()
            optimizer.zero_grad()

            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()

            tokens = tokenize(list(txt), context_length=CONTEXT_LENGTH)
            mask = torch.randn(BATCH_SIZE) < NULL_EMB_RATE
            tokens[mask] = NULL_TOKEN
            tokens = tokens.to(device=device)

            loss = get_loss(model, img, t, tokens, diffusion)
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            if step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                sample_plot_image(model, tokens[None, 0], diffusion, f"{epoch:03}(1)", caption=txt[0])
                sample_plot_image(model, tokens[None, 1], diffusion, f"{epoch:03}(2)", caption=txt[1])


def get_loss(model, x_0, t, tokens, diffusion):
    x_noisy, noise = diffusion.forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t, tokens)
    return F.l1_loss(noise, noise_pred)


@torch.no_grad()
def sample_plot_image(model, tokens, diffusion, filename, **kwargs):
    # model.eval()
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,8))
    plt.axis('off')
    # if torch.equal(tokens.detach().cpu(), NULL_TOKEN):
    #     plt.title("Null label")
    # else:
    #     plt.title(kwargs["caption"])

    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, tokens, model, diffusion)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    if filename is not None:
        plt.savefig(f"./outputs/diffusion/{filename}")
    plt.close()


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))


if __name__ == "__main__":
    diffusion = Diffusion(T)
    model = UNet(
        n_vocab=N_VOCAB,
        context_length=CONTEXT_LENGTH,
        transformer_width=TRANSFORMER_WIDTH,
        transformer_layers=TRANSFORMER_LAYERS,
        transformer_heads=TRANSFORMER_HEADS,
        qkv_heads=QKV_HEADS
    )
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    train_data, _ = load_data()
    dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    train(model, dataloader, diffusion)

