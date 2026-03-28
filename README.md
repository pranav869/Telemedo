# NOTSOMEONE
A real-time telemedicine platform featuring secure video consultations and live bi-directional English-Tamil speech translation with critical risk analysis.
# 🏥 Tele-Health AI & Smart Consultation Platform

A comprehensive Tele-Health web application integrating AI-powered disease prediction, real-time video consultations with translation, medical report analysis, and cancer screening tools.

## 🚀 Features

* *👨‍⚕️ Doctor & Patient Modes:* Distinct interfaces for medical professionals and patients.
* *📹 Video Consultation:* Real-time video calls using *Agora RTC, integrated with **Azure Speech AI* for live translation (English ↔ Tamil).
* *🧬 AI Cancer Screening:* Uses a custom *TensorFlow/VGG16* model to classify chest CT scan images (Adenocarcinoma, Large cell, Squamous cell, and Normal).
* *🤖 Disease Prediction (RAG):* diagnosing assistant using *Pinecone* (Vector DB) and *Groq (Llama 3)* to analyze symptoms against a trusted medical knowledge base.
* *📄 Medical Report Analyzer:* OCR-based analysis of medical reports using *API Ninjas* and LLMs to generate patient-friendly summaries and PDFs.
* *📍 Hospital Locator:* Intelligent routing to nearby hospitals using *OSRM* and *Leaflet Maps*.

## 🛠️ Tech Stack

* *Backend:* Python, Flask, Flask-SocketIO
* *Frontend:* HTML5, Bootstrap 5, JavaScript
* *AI/ML:* TensorFlow (Keras), LangChain, Groq (Llama 3), Pinecone
* *Real-Time:* Agora RTC, Azure Speech SDK, Socket.IO
* *Utilities:* Pandas, NumPy, FPDF2, Pillow

## ⚙️ Prerequisites

Before running the application, ensure you have the following installed:
* Python 3.9 or higher
* pip (Python package manager)

## 📦 Installation

1.  *Clone the Repository*
    bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    

2.  *Create a Virtual Environment (Recommended)*
    bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    

3.  *Install Dependencies*
    bash
    pip install -r requirements.txt
    

    Note: If you encounter an error regarding fpdf, ensure you uninstall fpdf and install fpdf2.

4.  *Model Setup*
    * Ensure the trained model file chest_cancer_classifier.h5 is placed in the root directory.

## wm️ Configuration (.env)

Create a .env file in the root directory and add your API keys.

```ini
# AI & LLM Keys
GROQ_API_KEY_NEW="your_groq_api_key"
API_NINJAS_KEY="your_api_ninjas_key"

# Database & Storage
PINECONE_KEY="your_pinecone_key"
JSONBIN_MASTER_KEY="your_jsonbin_key"
LOCALITY_BIN_ID="your_bin_id"

# Data Sources
HIDDEN_CSV_URL1="url_to_your_hospitals_csv"

# Note: Agora and Azure keys are currently configured in app.py or script tags
