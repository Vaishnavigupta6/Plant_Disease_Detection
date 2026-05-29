import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from model import PlantDiseaseCNN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
checkpoint = torch.load(
    r'C:\plant_disease_project\best_plant_model.pth',
    map_location=device
)
classes    = checkpoint['classes']
num_classes = len(classes)

# Load model
model = PlantDiseaseCNN(num_classes).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded! Trained val acc: {checkpoint['val_acc']:.4f}")
print(f"Classes: {num_classes}")

# Test transforms
test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset
full_dataset = datasets.ImageFolder("PlantVillageDataset/PlantVillage/", transform=test_transforms)

# Use last 500 samples as test set
import random
random.seed(42)
indices = random.sample(range(len(full_dataset)), 500)
test_set = Subset(full_dataset, indices)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

# Run testing
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# Accuracy
accuracy = (all_preds == all_labels).mean() * 100
print(f"\nTest Accuracy: {accuracy:.2f}%")

# Classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(r'C:\plant_disease_project\confusion_matrix.png')
plt.show()
print("\nConfusion matrix saved!")