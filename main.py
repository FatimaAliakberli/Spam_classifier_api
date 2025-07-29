from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import string
from wordcloud import STOPWORDS
import spacy

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load your trained model
model = joblib.load("spam_classifier.pkl")

# Preprocessing functions
def clean_test(s):
    for cs in s:
        if not cs in string.ascii_letters:
            s = s.replace(cs, ' ')
    return s.rstrip('\r\n')

def remove_little(s):
    wordsList = s.split()
    resultList = [element for element in wordsList if len(element) > 2]
    return ' '.join(resultList)

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in STOPWORDS]
    return " ".join(filtered_words)

def preprocess(text):
    text = clean_test(text)
    text = remove_little(text)
    text = lemmatize_text(text)
    text = remove_stopwords(text)
    return text

# Create FastAPI app
app = FastAPI()

# ✅ Add this block to allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with ["chrome-extension://<your-extension-id>"] if you want security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class EmailRequest(BaseModel):
    text: str

@app.post("/classify")
def classify_email(req: EmailRequest):
    processed_text = preprocess(req.text)
    prediction = model.predict([processed_text])[0]
    result = "HAM (Not Spam)" if prediction == 1 else "SPAM"
    return {"prediction": result}
