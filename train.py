import torch
import eff_train as etr
import eff_val as ev
import Checkpoint as cb
from processing_data import build_dataloader

EPOCHS = 20

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = build_dataloader()

    model = etr.model.to(device)
    criterion = etr.criterion
    optimizer = etr.optimizer

    best_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss = etr.train_one_epoch(
            model, train_loader, optimizer, criterion
        )

        val_loss, val_acc = ev.validate(
            model, val_loader, criterion
        )

        print(
            f"Epoch {epoch} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            f"Acc {val_acc:.4f}"
        )

        cb.save_checkpoint(epoch, model, optimizer, best_acc, "last.pt")

        if val_acc > best_acc:
            best_acc = val_acc
            cb.save_checkpoint(epoch, model, optimizer, best_acc, "best.pt")

        # Unfreeze backbone
        if epoch == 5:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

if __name__ == "__main__":
    main()
