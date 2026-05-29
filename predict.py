import torch
from torchvision import transforms
from PIL import Image
from model import PlantDiseaseCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
checkpoint  = torch.load(r'C:\plant_disease_project\best_plant_model.pth', map_location=device)
classes     = checkpoint['classes']
num_classes = len(classes)

model = PlantDiseaseCNN(num_classes).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image_path):
    image  = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs     = model(tensor)
        probs       = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)

    disease    = classes[predicted.item()]
    confidence = confidence.item() * 100

    return disease, confidence

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py leaf.jpg")
    else:
        disease, confidence = predict(sys.argv[1])
        print(f"Disease : {disease}")
        print(f"Confidence: {confidence:.2f}%")