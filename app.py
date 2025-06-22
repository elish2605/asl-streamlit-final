import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import os
import torch.nn as nn
model_path= "asl_resnet_model.pt"
# ---- Model Definition ----
class ASLResNet101(nn.Module):
    def __init__(self, num_classes=29):
        super(ASLResNet101, self).__init__()
        self.backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)

# ---- Load Model ----
def load_model(model_path):
    model = ASLResNet101(num_classes=29)
    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    return model

# ---- Image Preprocessing ----
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = image.convert('RGB')
    return transform(image).unsqueeze(0)

# ---- Predict ----
def predict(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# ---- Streamlit UI ----
st.title("ASL Image Sequence Predictor")

show_images = st.checkbox("Show uploaded images", value=False)
uploaded_files = st.file_uploader("Upload ASL letter images in order", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

model_path = "asl_resnet_model.pt"  # ✔️ Doit être placé dans le même dossier que ce fichier
class_names = [chr(ord('A') + i) for i in range(26)] + ['del', 'nothing', 'space']
model = load_model(model_path)

if uploaded_files:
    predictions = []

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        image_tensor = preprocess_image(image)
        prediction = predict(model, image_tensor, class_names)
        predictions.append(prediction)

    if show_images:
        cols = st.columns(len(uploaded_files))
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            with cols[idx]:
                st.image(image, use_container_width=True)

    st.markdown("### 🔤 Résultat :")
    st.markdown(" ".join(predictions))
