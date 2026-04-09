# 🏥 SkinCare AI: Disease Detection & LLM Advisor

An end-to-end AI-powered healthcare system designed to analyze skin images, classify diseases using deep learning ensembles, and provide intelligent medical recommendations via Large Language Models (LLM).

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED?style=flat&logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview
The **SkinCare AI** system evaluates the synergy between Computer Vision and Natural Language Processing in a clinical context. Users upload an image of a skin condition, which is then processed through a modular pipeline to provide real-time diagnostic insights and personalized health advice.

### Key Objectives:
*   **Disease Classification:** Accurate detection of skin conditions using state-of-the-art CNNs.
*   **LLM Integration:** Providing context-aware advice using Groq/Grok-powered LLMs.
*   **Modular Pipeline:** Seamless flow from image preprocessing to report generation.

---

## 🛠️ Technical Stack

| Component | Technology |
| :--- | :--- |
| **Backend** | Python, FastAPI, Uvicorn |
| **Frontend** | Streamlit |
| **Database** | MongoDB Atlas |
| **AI/ML** | TensorFlow, PyTorch, Ultralytics (YOLOv8) |
| **LLM** | Groq API (Llama 3 / Grok), OpenAI |
| **Reporting** | ReportLab (PDF Generation), Matplotlib, Plotly |
| **DevOps** | Docker, Docker Compose |

---

## 🧬 Model Architecture
This system utilizes a multi-model ensemble approach to ensure high confidence and accuracy:
*   **EfficientNetB0 & MobileNetV2:** Lightweight models optimized for real-time mobile/web performance.
*   **ResNet50:** Deep residual learning for high-precision feature extraction.
*   **YOLOv8:** Used for localized lesion detection and region-of-interest focus.

**Dataset:** [Kaggle Skin Disease Image Dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset) (containing 10 unique classes).

---

## 🚀 Getting Started

### Prerequisites
*   Docker & Docker Compose
*   Groq/OpenAI API Key

### Running with Docker (Recommended)
The easiest way to start the system is using Docker Compose, which orchestrates the Backend, Frontend, and Database.

```bash
# Clone the repository
git clone git@github.com:SifatSwapnil2022/SkinCareAI.git
cd SkinCareAI

# Build and Start
docker-compose up --build
```
---
### 🧬 Data Engineering & Model Architecture
Dataset & Class Balancing
The system is trained on the Kaggle Skin Disease Dataset, focusing on 10 distinct classes. To ensure the model performs equally well across rare and common diseases, a rigorous Data Resampling strategy was implemented:

1. Original Class Distribution:
The raw dataset was highly imbalanced, with Melanocytic Nevi (7,970 images) significantly outnumbering Atopic Dermatitis (1,257 images).

2. Resampling Strategy:

Oversampling: Used for minority classes (e.g., Eczema, Psoriasis, Fungal Infections) to reach a target of ~7,000 images per class.

Undersampling: Applied to the Melanocytic Nevi class (deleting 970 images) to prevent model bias toward benign moles.

Final Balanced Classes:

Eczema (~7k images)

Melanoma (~7k images)

Atopic Dermatitis (~7k images)

Basal Cell Carcinoma (BCC) (~7k images)

Melanocytic Nevi (NV) (~7k images)

Benign Keratosis-like Lesions (BKL) (~7k images)

Psoriasis & Lichen Planus (~7k images)

Seborrheic Keratoses (Benign Tumors) (~7k images)

Tinea Ringworm (Fungal Infections) (~7k images)

Warts & Viral Infections (~7k images)



---

### AI Pipeline

The modular pipeline ensures a multi-stage analysis:

Preprocessing: Image normalization and resizing via OpenCV.

Classification: Ensemble of EfficientNetB0, MobileNetV2, and ResNet50.

Detection: YOLOv8 for precise lesion localization.

Reasoning: The classification output is fed into the LLM Module (Grok/Llama 3) to generate human-readable explanations.

---

### 🛠️ Usage & Commands (Local Development)

```bash

# Run Backend (FastAPI)
uvicorn main:app --reload --host 127.0.0.1 --port 8001

# Run Frontend (Streamlit)
streamlit run app.py
```
### (Docker Management)

```bash

# Build and Start all services (Backend, Frontend, MongoDB)
docker-compose up --build

# Stop and Remove containers
docker-compose down

```

### 📄 Reporting

Generates PDF reports for each analyzed image using ReportLab, including:

- Detected condition
- Confidence score
- Visualizations via Matplotlib/Plotly
- LLM-based recommendations

  <img width="666" height="906" alt="image" src="https://github.com/user-attachments/assets/42ac6ca4-544c-483a-80a0-662e533147c0" />

  <img width="1192" height="722" alt="image" src="https://github.com/user-attachments/assets/1f0726c3-4d38-4410-91b1-b2cea4034b0e" />

---

### 🖥️ User Interface Screenshots


1️⃣ Landing Page (When User Opens SkinCare AI)


<img width="1886" height="804" alt="image" src="https://github.com/user-attachments/assets/6eecba49-2635-4991-9a9f-9f01183bbcb2" />


2️⃣ Sign In / Sign Up Page


<img width="1476" height="541" alt="image" src="https://github.com/user-attachments/assets/477b1841-0173-46d8-867a-363fce6b2b65" />


3️⃣ After Authentication (Main Dashboard Entry)


<img width="1882" height="931" alt="image" src="https://github.com/user-attachments/assets/d44304a2-f504-430d-b3b9-4d9aceb0412c" />


4️⃣ Model Selection Panel
(User can choose between four trained models)


<img width="752" height="300" alt="image" src="https://github.com/user-attachments/assets/14b60a01-c7db-4f43-9689-8c99077cba0c" />

5️⃣ Prediction Interface (Example: ResNet50 Selected)


<img width="1890" height="881" alt="image" src="https://github.com/user-attachments/assets/da8b97a9-b770-4b95-9382-8774291297b5" /> <img width="1633" height="437" alt="image" src="https://github.com/user-attachments/assets/041a1625-2388-4533-952c-48ebe0b50ad9" />


6️⃣ User Dashboard


<img width="1639" height="756" alt="image" src="https://github.com/user-attachments/assets/35604b01-4d8b-4964-ae71-1339e9f2a327" />


7️⃣ User History Page


<img width="1881" height="901" alt="image" src="https://github.com/user-attachments/assets/d5f908f8-2031-46dd-b704-4379dc2ff0da" />


8️⃣ User Profile Page


<img width="1087" height="328" alt="image" src="https://github.com/user-attachments/assets/4885efc1-2e1a-40d8-abc2-82c6d51f3872" />


---

### 📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.

---

### 👥 Owner & Developer
MD SIFATULLAH SHEIKH

Email: mdsifatullahsheikh@gmail.com
Website: https://myportfolio-bn96.onrender.com/

Project Role: System Architecture, AI/ML Pipeline, Backend & Frontend Development

---

### 🙏 Acknowledgments
Dataset: Kaggle Skin Disease Dataset by Ismail Promus.

Open Source Tools: FastAPI, Streamlit, Docker, TensorFlow/PyTorch, and Ultralytics (YOLOv8).

LLM Providers: Groq and OpenAI for enabling the intelligent medical reasoning module.

---
