# Kidney CT Disease Prediction App

This repository contains a deep learning–based application that predicts kidney disease from CT scans. The application classifies kidney CT images into one of four categories: **Cyst**, **Normal**, **Stone**, or **Tumor**. It consists of two main components:

1. **Prediction App:** A Streamlit web application that allows users to upload a Kidney CT image and get an immediate prediction along with a confidence score.
2. **Model Training Notebook:** A Jupyter Notebook (or training script) that demonstrates how to fine-tune a pre-trained EfficientViT model using a kidney CT dataset.

## Features

- **Image Upload & Prediction:**  
  Upload an image (jpg, jpeg, or png) via the Streamlit interface to predict the kidney disease type.
- **Deep Learning Inference:**  
  Leverages a fine-tuned EfficientViT model (via the `timm` library) and PyTorch for inference.

- **Interactive User Interface:**  
  Built using Streamlit, the interface displays the uploaded image, along with prediction results and confidence levels.

- **Model Training (Notebook):**  
  Provides a training pipeline that:
  - Loads a CT Kidney dataset from an ImageFolder structure.
  - Applies data augmentations and transforms.
  - Splits the data into training and validation sets.
  - Fine-tunes the EfficientViT model and saves the best model weights.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Jnan-py/kidney-disease-prediction.git
   cd kidney-disease-prediction
   ```

````

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Prediction App

1. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
2. **Upload an Image:**
   Supported formats: jpg, jpeg, png.
   The app will display the uploaded image and then output the predicted disease type (Cyst, Normal, Stone, or Tumor) along with a prediction confidence.

### Model Training

- Open the provided Jupyter Notebook (or training script) to see how the model is trained using the CT Kidney Dataset.
- The training process uses the `timm` library to load a pre-trained EfficientViT model, which is then fine-tuned on the dataset.
- The best model weights are saved to `efficientvit_m2_kidney_disease_classifier.pth` for later inference in the prediction app.

## Project Structure

```
kidney-disease-prediction/
│
├── app.py                                  # Streamlit prediction application
├── training_notebook.ipynb                 # Jupyter Notebook for model training (or equivalent Python script)
├── efficientvit_m2_kidney_disease_classifier.pth  # Saved model weights (generated after training)
├── README.md                               # Project documentation
└── requirements.txt                        # Python dependencies
```

## Technologies Used

- **Streamlit:** For building the interactive web interface.
- **PyTorch & Torchvision:** For model development, training, and inference.
- **timm:** For accessing pre-trained EfficientViT models.
- **Pillow:** For image processing.
- **NumPy & scikit-learn:** For data manipulation and evaluation.
- **tqdm:** For visualizing progress during training.

---

To run the prediction app, activate your virtual environment and execute:

```bash
streamlit run app.py
```

Feel free to adjust the documentation as needed. Enjoy building and exploring kidney disease predictions!
````
