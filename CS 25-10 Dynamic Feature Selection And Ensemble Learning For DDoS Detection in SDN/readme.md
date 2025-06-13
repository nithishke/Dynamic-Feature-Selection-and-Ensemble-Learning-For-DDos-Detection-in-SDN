# 🔐 Dynamic Feature Selection and Ensemble Learning for DDoS Detection in SDN

This project focuses on detecting Distributed Denial of Service (DDoS) attacks in **Software Defined Networks (SDN)** using intelligent machine learning techniques. It combines **dynamic feature selection**, **PCA for dimensionality reduction**, and an **ensemble learning model** to ensure fast and accurate detection.

---

## 🚀 How to Run This Project

### 📦 Requirements

Ensure the following are installed on your system:

- Python (3.8+)
- Flask
- scikit-learn
- pandas
- numpy
- VS Code (or any code editor)

Install dependencies:
```bash
pip install scikit-learn flask pandas numpy

```
📥 Setup Steps
Download the Project

Click the green Code button → Download ZIP.

Extract the ZIP file to your desired folder.

Download the Dataset

📁 [Download from Google Drive](https://drive.google.com/drive/folders/1KLoVqwKwvmXRDjUHAqiJRjzMYRkKNDa9?usp=sharing)

Place the dataset in the root folder of the project.

Train the ML Models by using the below Command
```bash
python train_model.py
```
This will:

Perform dynamic feature selection

Apply PCA

Train and save the ensemble classifiers

Run the Web App
```bash
python app.py
```
The app will start on http://localhost:5000/.

🧠 Model Details
Feature Selection: Cross Entropy with Mutual Information

Dimensionality Reduction: Principal Component Analysis (PCA)

Ensemble Model:

Bernoulli Naive Bayes

MLP Classifier

Passive-Aggressive Classifier

Stacking Classifier

🧪 Dataset
[CICDDoS2019](https://drive.google.com/drive/folders/1KLoVqwKwvmXRDjUHAqiJRjzMYRkKNDa9?usp=sharing) and other real-world SDN attack datasets

Includes simulated DDoS flows for training

🙋‍♂️ Author
Nithish Kumar K E
Passionate about Cybersecurity | Machine Learning | SDN

[LinkedIn Profile](https://www.linkedin.com/in/nithish-kumar-k-e/)

