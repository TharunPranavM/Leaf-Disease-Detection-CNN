
---

# 🌿 **Plant Disease Detection using EfficientCNN**

![Plant Disease Detection](https://img.shields.io/badge/Status-Completed-green) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Torch](https://img.shields.io/badge/Torch-1.9+-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-1.10+-red)

## 🚀 **Project Overview**
This project uses a **Convolutional Neural Network (CNN)** model, `EfficientCNN`, to detect plant diseases from images. The web-based interface, built using **Streamlit**, allows users to:
- 🌾 **Upload images** or capture them using a webcam.
- 🔍 **Predict plant diseases** with confidence scores.
- 📊 Display model accuracy and average loss metrics.

---

## 📁 **Folder Structure**
```
📂 Project Root
 ┣ 📜 app.py               → Streamlit web app for disease detection
 ┣ 📜 evaluate_model.py    → Script to evaluate the model and save metrics
 ┣ 📜 model.py             → CNN model architecture and training functions
 ┣ 📄 disease_info.csv     → Information on plant diseases
 ┣ 📄 evaluation_results.json → Model performance metrics (accuracy, loss)
 ┣ 📄 plant_disease_model_1_state_dict.pt → Trained model weights
 ┣ 📄 checkpoint.pth       → Model checkpoint during training
 ┗ 📄 README.md            → Project documentation
```

---

## 🔥 **Key Features**
✅ **Real-time Image Capture:** Use your webcam to capture and analyze images.  
✅ **Model Evaluation:** Display accuracy and loss metrics from test evaluations.  
✅ **Preprocessing Pipeline:** Resize, crop, normalize, and convert images to tensors.  
✅ **Web-Based Interface:** Built with Streamlit for an interactive experience.  

---

## ⚙️ **Workflow**
The project follows the below workflow:

### 1️⃣ **Data Preprocessing**
- The dataset is loaded using **PyTorch's `ImageFolder`**.  
- Images are resized to `224x224`, normalized, and converted into tensors.  
- The dataset is split into **training, validation, and test sets**.  
- **DataLoader** handles batching and shuffling.

### 2️⃣ **Model Training**
- The `EfficientCNN` model is trained using:
  - **CrossEntropyLoss** for multi-class classification.  
  - **Adam optimizer** with a learning rate of `0.001`.  
  - Model checkpointing saves weights periodically.  
- The training process uses:
  - **Batch Gradient Descent**
  - Epoch progress display with batch-wise updates.  
- The final model is saved as:
  - `plant_disease_model_1.pt` → Full model  
  - `plant_disease_model_1_state_dict.pt` → Model weights only  

### 3️⃣ **Model Evaluation**
- The model is evaluated using `evaluate_model.py`.  
- It calculates:
  - ✅ **Accuracy**
  - 📉 **Average Loss**  
- The evaluation results are saved in `evaluation_results.json`.

### 4️⃣ **Web Interface**
- The **Streamlit web app** uses:
  - **OpenCV** for image capture.  
  - **Pillow** for image processing.  
- Users can:
  - 📸 Capture images from the webcam.  
  - 📁 Upload images.  
- The app predicts the disease and displays:
  - 🌿 Disease name  
  - 🔬 Confidence score  
  - 📊 Model accuracy and average loss  

---

## 🛠️ **Tech Stack**
- **Framework:** Streamlit (for the web interface)  
- **Model:** PyTorch-based `EfficientCNN` with multiple convolutional and dense layers  
- **Dataset:** ImageFolder dataset structure with PyTorch's DataLoader  
- **Backend:** OpenCV for capturing images, PIL for image processing  

---

## 🚀 **Getting Started**

### 📦 **Installation**
1. Clone the repository:
```bash
git clone <repository_link>
cd <project_folder>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### ▶️ **Running the App**
Start the Streamlit application:
```bash
streamlit run app.py
```

---

## 🛠️ **Model Architecture**
The `EfficientCNN` model consists of:
- **3 Convolutional blocks** with BatchNorm, ReLU, and MaxPooling layers.
- **Dropout** to prevent overfitting.
- **Fully Connected (FC)** layers with 1024 neurons and output layer with `K` classes.
- **CrossEntropyLoss** for multi-class classification.

---

## 🖼️ **Usage Guide**

### 📸 **Image Upload or Capture**
- Upload images in **JPG, PNG, or JPEG** format.  
- Capture images using the webcam.  

### 🔍 **Disease Prediction**
- The app displays the detected **disease name** and **confidence score**.  
- View model accuracy and loss metrics with the "Show Model Accuracy" option.  

---

## 📊 **Evaluation Metrics**
After training, the model is evaluated using `evaluate_model.py`, which computes:
- ✅ **Accuracy**
- 📉 **Average Loss**

Metrics are saved to `evaluation_results.json`.

---

## ⚙️ **Configuration**
To modify parameters like batch size, epochs, or learning rate, edit the corresponding variables in:
- `model.py` → For model training and checkpointing.  
- `evaluate_model.py` → For testing and saving evaluation metrics.  

---

## 🔥 **Sample Output**
```
![image](https://github.com/user-attachments/assets/862ea043-780b-43b3-b5dd-9328ee897346)

🌱 Predicted Disease: Apple : Scab 
🔬 Confidence: 89.13%  
📊 Model Test Accuracy: 77.07%  
📉 Average Loss: 0.8361  
```

---
