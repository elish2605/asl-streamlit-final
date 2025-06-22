import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Recall, Precision

# --- GLOBAL CONSTANTS ---
ASL_CLASSES = [chr(ord('A') + i) for i in range(26)]
ASL_CLASSES.extend(['del', 'nothing', 'space'])
NUM_CLASSES_ASL = len(ASL_CLASSES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

# --- Model Definition ---
class WNBModule(pl.LightningModule):
    def __init__(self, num_classes: int = NUM_CLASSES_ASL, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.vgg19(pretrained=False)
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average='weighted')
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average='weighted')

    def forward(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def configure_optimizers(self):
        return None

    def training_step(self, batch, batch_idx):
        return None

    def validation_step(self, batch, batch_idx):
        return None

    def test_step(self, batch, batch_idx):
        return None

    def unfreeze_all_layers(self):
        pass

# --- Model Path ---
MODEL_STATE_DICT_PATH = "VGG19.pt"

# --- Load Model ---
@st.cache_resource
def load_model_state_dict(model_state_dict_path, num_classes):
    if not os.path.exists(model_state_dict_path):
        st.error(f"Error: Model file not found: {model_state_dict_path}")
        st.stop()
        return None

    st.write(f"Loading TorchScript model: {model_state_dict_path}")
    try:
        model = torch.jit.load(model_state_dict_path, map_location=torch.device('cpu'))
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()
            st.write("Model loaded on GPU.")
        else:
            model = model.cpu()
            st.write("Model loaded on CPU.")

        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
        return None

model = load_model_state_dict(MODEL_STATE_DICT_PATH, NUM_CLASSES_ASL)

# --- Image Transforms ---
inference_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# --- Predict on Single Image ---
def predict_on_image(image_pil):
    if model is None:
        return "Model not loaded.", "N/A"

    try:
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        image_tensor = inference_transforms(image_pil).unsqueeze(0)

        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        else:
            image_tensor = image_tensor.cpu()

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        conf, predicted_class_idx = torch.max(probabilities, 1)
        predicted_class = ASL_CLASSES[predicted_class_idx.item()]
        confidence = conf.item() * 100

        return predicted_class, f"{confidence:.2f}%"

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "?", "N/A"

# --- UI ---
st.title("🤟 ASL Word Translator")
st.markdown("Upload one or more **ASL letter images** (A–Z). The model will predict each and show the final word.")

uploaded_files = st.file_uploader("📁 Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    predictions = []
    confidences = []
    show_images = []

    with st.spinner("🪄 Predicting letters..."):
        progress = st.progress(0)
        total = len(uploaded_files)

        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            predicted_label, confidence = predict_on_image(image)
            if len(predicted_label) == 1:
                predictions.append(predicted_label)
                confidences.append(confidence)
                show_images.append(image)

            progress.progress((idx + 1) / total)

    if not any(st.session_state.get(f"img_{i}", False) for i in range(len(predictions))):
        st.toast("✅ Prediction complete!", icon="👏")

    full_word = ''.join(predictions)

    st.subheader("🔤 Predicted Word:")
    st.success(f"**{full_word}**")

    st.subheader("🧠 Letter Predictions:")
    for i, (letter, conf) in enumerate(zip(predictions, confidences)):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{letter}** — Confidence: {conf}")
        with col2:
            if st.checkbox(f"Show", key=f"img_{i}"):
                st.image(show_images[i], caption=f"Predicted: {letter} | Confidence: {conf}", width=150)

# --- Sidebar Info ---
st.sidebar.header("ℹ️ About")
st.sidebar.info(
    "This app uses a fine-tuned VGG19 model to recognize American Sign Language 🤟 (ASL) letters.\n\n"
    "- Upload one image per letter\n"
    "- Order matters (first letter uploaded = first letter predicted)\n"
)
st.sidebar.write(f"Supported classes: {NUM_CLASSES_ASL}")
