## Pipeline chuẩn train efficientnet-b3 (300 × 300 × 3)

    Load ảnh
    → Resize + Normalize
    → Augment train
    → Batch loader
    → Freeze backbone
    → Train head
    → Unfreeze 1–2 block
    → Fine-tune
    → Monitor val loss/acc
    → Save best model
    → Test & compute confusion matrix


## Handle dataset 
    Ảnh gốc
    → kiểm tra RGB
    → chia folder theo class
    → chia train / val / test
    → resize 300×300
    → normalize ImageNet
    → augment (chỉ train) -> lá chụp không có nhiều gốc như trong thực tế, nên cần Augmentation mạnh về độ sáng và độ tương phản
    → đưa vào model
    ![img sau handle](image-1.png)

## Train 
for epoch:
    model.train()
        forward
        loss
        backward
        optimizer.step()

    model.eval()
        forward (no grad)
        val loss / acc

    save checkpoint

## Tư duy

- Dùng pretrained ImageNet

- Thay classifier cuối theo số class dataset

- Giai đoạn đầu: freeze backbone

![So sánh pytorch và timm](image.png)

### Dataset
- Link: https://drive.google.com/file/d/1woZmhlRF15m5CEuNAV2G8XrJfxODF1t1/view?usp=sharing