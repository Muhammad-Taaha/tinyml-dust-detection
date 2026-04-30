import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mcunet.model_zoo import build_model
import os

# ---------- Configuration ----------
net_id = "mcunet-in0"          # Valid ID from the list
input_size = 96                # Force square input (model expects 96x96)
data_root = "./cropped_dataset"
batch_size = 32
num_epochs = 30
device = torch.device('cpu')

# ---------- Load model architecture (without pretrained weights) ----------
model, _, _ = build_model(net_id=net_id, pretrained=False)

# Manually load pre‑trained weights if they exist
ckpt_path = os.path.expanduser(f"~/.torch/mcunet/{net_id}.pth")
if os.path.exists(ckpt_path):
    print(f"Loading pre‑trained weights from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
else:
    print("No pre‑trained weights found – training from scratch (may converge poorly).")

# Replace classifier head for 2 output classes
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, 2)
model = model.to(device)

# ---------- Data transforms with FIXED square resize ----------
transform_train = transforms.Compose([
    transforms.Resize((input_size, input_size)),   # force 96x96
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((input_size, input_size)),   # same fixed size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------- Load datasets ----------
train_ds = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=transform_train)
val_ds   = datasets.ImageFolder(os.path.join(data_root, 'val'),   transform=transform_val)

# Use num_workers=0 to avoid multiprocessing issues
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_ds)}")
print(f"Validation samples: {len(val_ds)}")
print(f"Classes: {train_ds.classes}")

# ---------- Optimizer & loss ----------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# ---------- Training loop ----------
best_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    acc = 100 * correct / total
    print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), f"best_{net_id}_solar.pth")
        print(f"  -> saved best model (acc={best_acc:.2f}%)")

print(f"Fine-tuning complete. Best validation accuracy: {best_acc:.2f}%")