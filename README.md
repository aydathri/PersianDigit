# Persian Handwritten Digit Recognition (SVM)

This is a comprehensive Machine Learning project focused on the 
**classification of handwritten Persian digits (0-9)**.
It utilizes a **Support Vector Machine (SVM)** classifier, trained on the HODA dataset,
and features a complete end-to-end pipeline, from data preprocessing and hyperparameter tuning to a
live web application demo built with **Streamlit**.

## Key Features

* **Classification Core:** Robust SVC model using the RBF (Radial Basis Function) kernel.
* **Pipeline:** Dedicated scripts for training (`train_model.py`) and live testing (`test_real_data.py`).
* **Preprocessing:** Advanced OpenCV techniques (contour detection, smart cropping, centering) ensure high accuracy even with varied real-world image inputs.
* **Evaluation:** Includes routines for calculating metrics, generating a Confusion Matrix, and plotting performance.
* **Real-Time Demo:** Interactive web application (`app.py`) for instantaneous prediction from uploaded images.

---

# Project Structure
```text
PersianDigits/
│
├── dataset/ # Contains the HODA dataset file (Data_hoda_full.mat)
├── digits_photo/ # Sample images for testing the model
├── plots/ # Output directory for generated performance charts (Confusion Matrix, Metrics)
├── app.py # Streamlit Web Application (Frontend Demo)
├── load_data.py # Data loading, splitting, and image preprocessing functions
├── train_model.py # Main script for Training, GridSearch, and Evaluation (The ML pipeline)
├── requirements.txt # Project dependencies
├── svm_digit_model.joblib # The final trained model file
├── test_real_data.py # Script to test the model on local images
└── README.md
```

---

# Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

You need **Python 3.8+** and the necessary libraries. All dependencies are listed in `requirements.txt`:

```bash
# Ensure your virtual environment is active
  pip install -r requirements.txt
```

---

# Installation
Clone the repository:

git clone https://github.com/aydathri/PersianDigit.git

cd PersianDigit
Dataset Requirement: Place the HODA dataset file (Data_hoda_full.mat) inside the dataset/ directory.

---

# Training and Evaluation
The core machine learning pipeline is executed via train_model.py.

1. Training the Model
Run this script to train the SVC model, print evaluation metrics, and save the final model file.

```bash
  python train_model.py
```

Output: The script will print the training and testing metrics, display performance plots, and save the final trained model to svm_digit_model.joblib.

2. Testing on Local Images
Use this script to test the model's performance on local images (e.g., from your digits_photo/ folder) before deploying the web app.

```bash
  python test_real_data.py
```

---

# Real-Time Web Application (Streamlit)
Test the deployed model interactively using the Streamlit app.

Requirement: Ensure the trained model file (svm_digit_model.joblib) exists in the root directory.

Launch the App:

```bash
  streamlit run app.py
```

The application will open in your browser (usually at http://localhost:8501).
Upload a PNG or JPG image of a handwritten Persian digit to see the model's prediction.

---

Expected PerformanceThe fully optimized model typically achieves high classification accuracy on the
HODA dataset:MetricScore
- (Example)Accuracy[98.XX%]
- Precision (Macro)[98.XX%]
- Recall (Macro)[98.XX%]
- F1-Score (Macro)[98.XX%]


The model was trained on [50,000] samples and tested on [10,000] samples at a [25x25] resolution.