import torch
def save_checkpoint(epoch, model, optimizer, best_acc, path):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_acc": best_acc
    }, path)
