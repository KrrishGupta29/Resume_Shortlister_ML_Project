from flask import Flask, render_template, request
import pickle
from PyPDF2 import PdfReader
import os

app = Flask(__name__)

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "resume" not in request.files:
            return "No file uploaded", 400
        
        file = request.files["resume"]
        if file.filename == "":
            return "No file selected", 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Extract text
        resume_text = extract_text_from_pdf(file_path)

        # Vectorize & predict
        vectorized_text = vectorizer.transform([resume_text])
        prediction = model.predict(vectorized_text)[0]

        result = "Relevant ✅" if prediction == 1 else "Not Relevant ❌"

        return render_template("result.html", result=result, resume_text=resume_text)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
