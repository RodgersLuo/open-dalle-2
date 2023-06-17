import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import yaml
import wandb

from diffusion import Diffusion
from prior import Prior

import sys
sys.path.insert(0, 'dataset')
from dataset import load_data

sys.path.insert(0, 'nn_components')
from tokenizer import tokenize

sys.path.insert(0, 'clip')
from model import CLIP

with open('./model_config.yml', 'r') as file:
    config = yaml.safe_load(file)
    prior_config = config["Prior"]
    clip_config = config["CLIP"]

wandb.init(
    # set the wandb project where this run will be logged
    project="open-dalle-2-prior",
    # track hyperparameters and run metadata
    config=config,
    # mode="disabled"
)

T = prior_config["diffusion_timesteps"]
BATCH_SIZE = prior_config["batch_size"]
IMG_SIZE = config["img_size"]
EPOCHS = prior_config["epochs"]
LR = prior_config["lr"]
CLIP_CONTEXT_LEN = clip_config["context_length"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

torch.cuda.empty_cache()

def train(prior: Prior, train_dataloader: DataLoader, val_dataloader: DataLoader, diffusion: Diffusion, clip: CLIP=None):
    prior.to(device)
    wandb.watch(prior, log="all", log_freq=50)

    optimizer = Adam(prior.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        for step, (img, txt, clip_embeds) in enumerate(train_dataloader):
            # txt = [classes[i] for i in txt]

            prior.train()
            optimizer.zero_grad()

            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()

            # tokenize text captions
            tokens = tokenize(list(txt), context_length=CLIP_CONTEXT_LEN)

            # img = img.to(device=device)
            tokens = tokens.to(device=device)

            image_embedding = clip_embeds["image_embedding"].to(device=device)
            text_embedding = clip_embeds["text_embedding"].to(device=device)
            text_encoding = clip_embeds["text_encoding"].to(device=device)

            # # obtain CLIP emmeddings
            # image_embedding = clip.encode_image(img)
            # text_embedding = clip.encode_text(tokens)
            # clip_embedding /= clip_embedding.norm(dim=1, keepdim=True)

            img_emb_noisy, noise = diffusion.forward_diffusion_sample(image_embedding, t, device)
            img_emb_pred = prior(img_emb_noisy, t, text_embed=text_embedding, text_encodings=text_encoding)
            loss = F.mse_loss(img_emb_pred, image_embedding)

            loss.backward()
            # torch.nn.utils.clip_grad.clip_grad_norm_(prior.parameters(), GRAD_CLIP)
            optimizer.step()

            if step == 0:
                baseline_sim = F.cosine_similarity(text_embedding, image_embedding, dim=-1).mean().item()
                predicted_sim = F.cosine_similarity(text_embedding, img_emb_pred, dim=-1).mean().item()
                images_sim = F.cosine_similarity(image_embedding, img_emb_pred, dim=-1).mean().item()

                random_sim = F.cosine_similarity(text_embedding[torch.randperm(BATCH_SIZE)], img_emb_pred, dim=-1).mean().item()
                print(f"Epoch {epoch} | Loss: {loss.item()}, Baseline Sim: {baseline_sim}, Predicted Sim: {predicted_sim}, Image sim: {images_sim}, Random Sim: {random_sim}")
                wandb.log({"train_loss": loss.item(), "baseline_sim": baseline_sim, "train_predicted_sim": predicted_sim, "train_image_sim": images_sim, "random_sim": random_sim})

                if epoch % 10 == 0 or epoch == EPOCHS - 1:
                    validate(prior, val_dataloader, diffusion, full_sample=True)
                else:
                    validate(prior, val_dataloader, diffusion, full_sample=False)

        if epoch == EPOCHS - 1:
            a = 1

def validate(prior: Prior, val_dataloader: DataLoader, diffusion: Diffusion, full_sample=False):
    prior.eval()  # set model to evaluation mode
    with torch.no_grad():
        img, txt, clip_embeds = next(iter(val_dataloader))
        # tokenize text captions
        tokens = tokenize(list(txt), context_length=CLIP_CONTEXT_LEN)

        tokens = tokens.to(device=device)
        image_embedding = clip_embeds["image_embedding"].to(device=device)
        text_embedding = clip_embeds["text_embedding"].to(device=device)
        text_encoding = clip_embeds["text_encoding"].to(device=device)

        if full_sample:
            img_emb_pred = prior.sample(text_embedding, text_encodings=text_encoding)
            predicted_sim = F.cosine_similarity(text_embedding, img_emb_pred, dim=-1).mean().item()
            images_sim = F.cosine_similarity(image_embedding, img_emb_pred, dim=-1).mean().item()
            loss = F.mse_loss(img_emb_pred, image_embedding).item()

            print(f"Full sample validation Loss: {loss}, Predicted sim: {predicted_sim}, Image sim: {images_sim}")
            wandb.log({"full_sampmle_val_loss": loss, "val_predicted_sim": predicted_sim, "val_image_sim": images_sim})
        else:
            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()

            img_emb_noisy, noise = diffusion.forward_diffusion_sample(image_embedding, t, device)
            img_emb_pred = prior(img_emb_noisy, t, text_embed=text_embedding, text_encodings=text_encoding)
            predicted_sim = F.cosine_similarity(text_embedding, img_emb_pred, dim=-1).mean().item()
            images_sim = F.cosine_similarity(image_embedding, img_emb_pred, dim=-1).mean().item()
            loss = F.mse_loss(img_emb_pred, image_embedding).item()
            print(f"Validation Loss: {loss}, Predicted sim: {predicted_sim}, Image sim: {images_sim}")
            wandb.log({"val_loss": loss, "val_predicted_sim": predicted_sim, "val_image_sim": images_sim})


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)

    # Create diffusion
    diffusion = Diffusion(T, schedule=prior_config["noise_schedule"])

    # Create Prior
    prior = Prior(
        clip_emb_dim=clip_config["embed_dim"],
        T=T,
        diffusion=diffusion,
        clip_context_len=CLIP_CONTEXT_LEN,
        clip_token_dim=clip_config["transformer_width"],
        xf_layers=prior_config["xf_layers"],
        # dim_head=prior_config["dim_per_head"],
        xf_heads=prior_config["xf_heads"],
        # vocab_size=clip_config["vocab_size"],
        # ff_mult=prior_config["ff_mult"],
    )
    param_size = 0
    for param in prior.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in prior.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    print("Num params: ", sum(p.numel() for p in prior.parameters()))

    # Load pretrained CLIP model
    clip = CLIP(
        embed_dim=clip_config["embed_dim"],
        image_resolution=IMG_SIZE,
        vision_layers=clip_config["vision_layers"],
        vision_width=clip_config["vision_width"],
        vision_patch_size=clip_config["vision_patch_size"],
        context_length=clip_config["context_length"],
        vocab_size=clip_config["vocab_size"],
        transformer_width=clip_config["transformer_width"],
        transformer_heads=clip_config["transformer_heads"],
        transformer_layers=clip_config["transformer_layers"]
    )
    clip.to(device="cpu")
    clip_state = torch.load(clip_config["model_path"])
    clip.load_state_dict(clip_state["model"])

    # Freeze CLIP model
    clip.eval()
    for param in clip.parameters():
        param.requires_grad = False

    train_data, val_data = load_data(img_size=IMG_SIZE,
                                     root_dir=config["data_path"],
                                     clip=clip, context_length=CLIP_CONTEXT_LEN,
                                     normalize_clip_embeddings=prior_config["normalize_clip_embeddings"])
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    train(prior, train_dataloader, val_dataloader, diffusion, clip=clip)
    state = {
        "model": prior.state_dict(),
        "config": config,
    }
    torch.save(state, prior_config["model_path"])
    # torch.save(prior.state_dict(), prior_config["model_path"])
