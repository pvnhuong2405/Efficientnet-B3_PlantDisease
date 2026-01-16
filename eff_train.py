import torch
from processing_data import build_dataloader
import timm
import torch.nn as nn

train_loader, val_loader = build_dataloader()

NUM_CLASSES = len(train_loader.dataset.dataset.classes)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = timm.create_model(
    "efficientnet_b3",
    pretrained=True,
    num_classes=NUM_CLASSES
)



model = model.to(device)


# Freeze backbone (giai đoạn warm-up)
for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

# Loss & Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

# Training Loop
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

