import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


relevant_resumes = [
    "Python developer with 3 years of experience in Django and Flask web frameworks.",
    "Data scientist skilled in Python, pandas, NumPy, and machine learning algorithms.",
    "Backend engineer with expertise in Python, REST API development, and SQL databases.",
    "Software engineer experienced in Python scripting and automation.",
    "Machine learning engineer proficient in scikit-learn, TensorFlow, and Python.",
    "Python developer with cloud deployment experience using AWS and Docker.",
    "AI developer using Python for natural language processing and computer vision.",
    "Full-stack developer with Python, React.js, and PostgreSQL experience.",
    "Data analyst skilled in Python, data visualization, and statistical modeling.",
    "Automation engineer using Python to build testing frameworks.",
    "Python programmer for IoT applications with Raspberry Pi.",
    "Python developer experienced in ETL pipelines and big data processing.",
    "Backend API developer using Python FastAPI and Flask.",
    "Data engineer using Python, Apache Spark, and Hadoop.",
    "Python developer with experience in machine learning competitions.",
    "Software engineer skilled in Python and unit testing frameworks.",
    "Python developer working on recommendation systems.",
    "Research assistant using Python for data analysis and simulation.",
    "Python developer with GraphQL API integration experience.",
    "Web scraping expert using Python BeautifulSoup and Scrapy."
]

not_relevant_resumes = [
    "Marketing specialist experienced in social media campaigns.",
    "Sales manager with strong negotiation and client acquisition skills.",
    "Mechanical engineer skilled in CAD and SolidWorks.",
    "Civil engineer experienced in structural analysis.",
    "Graphic designer with expertise in Photoshop and Illustrator.",
    "Electrical engineer working on embedded systems and PCB design.",
    "Content writer with SEO and blog writing experience.",
    "Business analyst skilled in finance and Excel modeling.",
    "HR manager with experience in recruitment and payroll.",
    "Event planner with strong organizational skills.",
    "Biomedical researcher with lab experiment expertise.",
    "Fashion designer experienced in clothing production.",
    "Customer support executive with CRM knowledge.",
    "Video editor skilled in Premiere Pro and After Effects.",
    "Architect with AutoCAD and 3D rendering skills.",
    "Teacher experienced in online tutoring platforms.",
    "Lawyer specialized in corporate law and legal compliance.",
    "Accountant with QuickBooks and tax filing experience.",
    "Chef experienced in gourmet cooking and kitchen management.",
    "Fitness trainer with nutrition planning experience."
]


all_resumes = relevant_resumes * 2 + not_relevant_resumes * 2 
labels = [1]*len(relevant_resumes)*2 + [0]*len(not_relevant_resumes)*2


combined = list(zip(all_resumes, labels))
random.shuffle(combined)
all_resumes, labels = zip(*combined)


df = pd.DataFrame({
    "resume_texts": all_resumes,
    "label": labels
})


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df["resume_texts"])
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model & vectorizer saved!")
