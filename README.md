# 📝 Machine Learning Based Resume Shortlister

A smart resume screening system that automatically classifies resumes as Relevant or Non-Relevant using a Machine Learning model trained with Scikit-Learn.
This project provides a simple and intuitive web interface built using Flask, where users can upload a PDF resume and instantly receive its prediction.

## 🚀 Features
✅ Machine Learning

-> Uses Scikit-Learn for ML pipeline

-> Logistic Regression model trained on labeled resume text

->Text extraction + preprocessing + vectorization

->Predicts whether a resume is Relevant or Non-Relevant

## 📂 Project Structure
```txt
Resume_Shortlister_ML_Project/
│
├── app.py                # Flask backend
├── model.pkl             # Trained ML model
├── vectorizer.pkl        # TF-IDF vectorizer
│
├── templates/
│   ├── index.html        # Upload page
│   └── result.html       # Prediction result page
│
├── static/               # (Optional) CSS/images if added
│
└── README.md             # Project documentation
```


## 🛠️ Technologies Used
| Component          | Technology                                            |
| ------------------ | ----------------------------------------------------- |
| Machine Learning   | Scikit-Learn, Logistic Regression                     |
| Feature Extraction | TF-IDF Vectorizer                                     |
| Backend            | Flask                                                 |
| Frontend           | HTML, CSS                                             |
| File Handling      | PyPDF / pdfminer / fitz (depending on implementation) |

## 📦 Installation & Setup
1️⃣ Clone the repository
```bash
git clone https://github.com/KrrishGupta29/Resume_Shortlister_ML_Project.git
cd Resume_Shortlister_ML_Project
```
2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
3️⃣ Run the Flask app
```bash
python app.py
```
4️⃣ Open in browser
```bash
http://127.0.0.1:5000/
```

