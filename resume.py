import torch

def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["best_acc"]
