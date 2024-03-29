#torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#other
import numpy as np
from model import CLIP
import os
import yaml
import wandb

import sys
sys.path.insert(0, "dataset")
sys.path.insert(0, "nn_components")
from dataset import load_data
from tokenizer import tokenize

with open("./model_config.yml", "r") as file:
    config = yaml.safe_load(file)
    clip_config = config["CLIP"]

wandb.init(
    # set the wandb project where this run will be logged
    project="open-dalle-2-clip",
    # track hyperparameters and run metadata
    config=config,
    # mode="disabled"
)

# Hyperparameters
IMG_SIZE = config["img_size"]
BATCH_SIZE = clip_config["batch_size"]
EPOCHS = clip_config["epochs"]
LR = clip_config["lr"]


# set the seed for reproducibility
rng_seed = 90
torch.manual_seed(rng_seed)

USE_GPU = True
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)


def load_dataset(batch_size=BATCH_SIZE, root_dir="./data"):
    # transform = transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #         ]
    #     )

    # train_img_dir = os.path.join(root_dir, "train/images")
    # train_table_dir = os.path.join(root_dir, "train/data.csv")
    # test_img_dir = os.path.join(root_dir, "test/images")
    # test_table_dir = os.path.join(root_dir, "test/data.csv")

    # print(train_img_dir)

    # train_dataset = ImageCaptionDataset(train_img_dir, train_table_dir, transform=transform)
    # test_dataset = ImageCaptionDataset(test_img_dir, test_table_dir, transform=transform)

    train_dataset, test_dataset = load_data(img_size=IMG_SIZE, root_dir=config["data_path"])

    captions = train_dataset.captions.unique()

    # Create train val split
    n = len(train_dataset)
    n_val = int(n/10)

    train_set, val_set = torch.utils.data.random_split(train_dataset, [n-n_val, n_val])
    print(len(train_set), len(val_set), len(test_dataset))

    loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    loader_val = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    loader_test = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    return loader_train, loader_val, loader_test, captions


def check_accuracy(loader, model, captions, analysis=False):
    # function for test accuracy on validation and test set

    num_top1_correct = 0
    num_top5_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for t, (x, y) in enumerate(loader):
            x = x.to(device=device, dtype=dtype)  # move to device
            # y = tokenize(list(y), context_length=model.context_length).to(device=device)
            y = np.array(list(y))
            captions = y
            # captions_tk = tokenize(y, context_length=model.context_length).to(device=device)
            captions_tk = tokenize(captions, context_length=model.context_length).to(device=device)

            # ims, txt = model(x, y)
            # ground_truth = torch.arange(len(ims)).long().to(ims.device)
            # loss = (F.cross_entropy(ims, ground_truth) + F.cross_entropy(ims.t(), ground_truth)).div(2)

            ims, txt = model(x, captions_tk)

            ground_truth = torch.arange(len(ims)).long().to(ims.device)
            loss = (F.cross_entropy(ims, ground_truth) + F.cross_entropy(ims.t(), ground_truth)).div(2)

            image_features = model.encode_image(x)
            text_features = model.encode_text(captions_tk)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=1)

            baseline_sim = F.cosine_similarity(text_features, image_features, dim=-1).mean().item()
            random_sim = F.cosine_similarity(text_features[torch.randperm(BATCH_SIZE)], image_features, dim=-1).mean().item()

            top1_value, top1_index = similarity.topk(1, dim=1)
            top5_values, top5_indices = similarity.topk(5, dim=1)

            # print(x.size())
            # print(image_features.size())
            # print(text_features.size())
            # print(similarity.size())

            # print(top1_index.size())
            # print(top5_indices.size())

            for i in range(x.size(0)):
                if y[i] == captions[top1_index[i, 0].cpu().detach().numpy()]:
                    num_top1_correct += 1
                if y[i] in captions[top5_indices[i].cpu().detach().numpy()]:
                    num_top5_correct += 1

            num_samples += x.size(0)
            # if t == 0 and analysis:
            #   stack_labels = y
            #   stack_predicts = preds
            # elif analysis:
            #   stack_labels = torch.cat([stack_labels, y], 0)
            #   stack_predicts = torch.cat([stack_predicts, preds], 0)
        top1_acc = float(num_top1_correct) / num_samples
        top5_acc = float(num_top5_correct) / num_samples
        print("Got %d / %d correct of val set, top 1 acc : %.2f, top 5 acc: %.2f, val loss: %.2f"
                % (num_top1_correct, num_samples, 100 * top1_acc, 100 * top5_acc, loss.item()))
        print(f"    Baseline similarity {baseline_sim:.3f}, random similarity {random_sim:.3f}")
        wandb.log({"top1_acc": top1_acc * 100, "top5_acc": top5_acc * 100, "val_loss": loss.item()})
        wandb.log({"baseline_sim": baseline_sim, "random_sim": random_sim})
        # if analysis:
        #   print("check acc", type(stack_predicts), type(stack_labels))
        #   confusion(stack_predicts, stack_labels)
        #   incorrect_preds(preds, y, x)
        return top1_acc, top5_acc


def train_part(model, optimizer, loader_train, loader_val, captions, epochs):
    """
    Train a model using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    print_every = 10

    model = model.to(device=device)  # move the model parameters to CPU/GPU

    wandb.watch(model, log="all", log_freq=25)

    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            # y = y.to(device=device)
            y = tokenize(list(y), context_length=model.context_length).to(device=device)

            ims, txt = model(x, y)

            # image_logits = ims @ txt.t() * model.logit_scale.exp()
            ground_truth = torch.arange(len(ims)).long().to(ims.device)
            loss = (F.cross_entropy(ims, ground_truth) + F.cross_entropy(ims.t(), ground_truth)).div(2)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            loss.backward()

            # Update the parameters of the model using the gradients
            optimizer.step()

            if t % print_every == 0:
                print("Epoch: %d, Iteration %d, loss = %.4f" % (e, t, loss.item()))
                wandb.log({"train_loss": loss.item()})
        check_accuracy(loader_val, model, captions)


if __name__ == "__main__":
    loader_train, loader_val, loader_test, captions = load_dataset()

    model = CLIP(
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

    # check_accuracy(loader_val, model, captions)


    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print("model size: {:.3f}MB".format(size_all_mb))

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-8)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: {}".format(params))

    train_part(model, optimizer, loader_train, loader_val, captions, epochs = EPOCHS)
    check_accuracy(loader_test, model, captions=captions)

    state = {
        "model": model.state_dict(),
        "config": config,
    }
    torch.save(state, clip_config["model_path"])

