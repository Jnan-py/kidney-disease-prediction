import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import timm

classes = ["Cyst", "Normal", "Stone", "Tumor"]

st.set_page_config(page_icon=":medical", page_title="Kidney Disease Prediction", layout="wide")

with st.sidebar:
    st.subheader("Dataset")
    with st.expander("Dataset Links", expanded=True):
        st.markdown("Dataset: [Kidney Disease Classification](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone/code)")

@st.cache_resource()
def load_model(model_name, num_classes, device):
    model = timm.create_model(
        model_name, pretrained=False, num_classes=num_classes)
    model_path = 'efficientvit_m2_kidney_disease_classifier.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model("efficientvit_m2",4,device)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                         std=[0.229, 0.224, 0.225])   
])


st.title("Kidney CT Disease Prediction")
st.write("Upload a Kidney CT image (jpg, jpeg, or png) to predict the disease type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:    
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", width = 400)
    
    image_tensor = transform(image).unsqueeze(0)  

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        prob, pred_idx = torch.max(probabilities, dim=1)

    pred_class = classes[pred_idx.item()]

    with col2:
        st.write(f"**Predicted Disease Type:** {pred_class}")
        st.write(f"**Prediction Confidence:** {prob.item()*100:.2f}%")
