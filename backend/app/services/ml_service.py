import pickle
from pathlib import Path
from app.utils.config import MODEL_PATH
from textblob import TextBlob
import re

def clean_text(text):
    # remove mentions but keep sentence structure
    text = re.sub(r'@\w+', ' ', text)
    
    # remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    
    # keep letters (upper + lower), numbers, and basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)
    
    # normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

class MLService:
    def __init__(self):
        self.model_bundle = None
        self.load_model()

    def load_model(self):
        model_file = Path(MODEL_PATH)

        if not model_file.exists():
            print("Model file not found")
            return False

        with open(model_file, "rb") as f:
            self.model_bundle = pickle.load(f)

        print("TF-IDF model loaded successfully")
        return True

    def predict(self, text: str) -> dict:
        if self.model_bundle is None:
            return {
                "label": "ERROR",
                "confidence": 0.0,
                "error": "Model not loaded"
            }

        try:
            vectorizer = self.model_bundle["vectorizer"]
            model = self.model_bundle["model"]

            # 🔥 KEY CHANGE: TF-IDF transform
            processed = clean_text(text)
            X = vectorizer.transform([text])

            prediction = model.predict(X)[0]

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                confidence = float(max(probs))
            else:
                confidence = 1.0

            return {
                "label": "urgent" if prediction == model.classes_[1] else "normal",
                "confidence": round(confidence, 3)
            }

        except Exception as e:
            return {
                "label": "ERROR",
                "confidence": 0.0,
                "error": str(e)
            }


ml_service = MLService()