"""ML model service for loading and using sklearn classifier"""
import pickle
from pathlib import Path
from app.utils.config import MODEL_PATH
from app.services.feature_extraction import extract_features


class MLService:
    """Service for loading and using trained ML models"""

    def __init__(self):
        """Load the pretrained model"""
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the sklearn model from disk"""
        model_file = Path(MODEL_PATH)
        
        if not model_file.exists():
            print(f"Warning: Model file not found at {MODEL_PATH}")
            print("Create a trained model and save it with pickle:")
            print("  import pickle")
            print("  pickle.dump(model, open('model.pkl', 'wb'))")
            return False
        
        try:
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict(self, text: str) -> dict:
        """
        Make a prediction using the trained model.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with 'label' and 'confidence'
        """
        if self.model is None:
            return {
                "label": "UNKNOWN",
                "confidence": 0.0,
                "error": "Model not loaded"
            }
        
        try:
            # Extract features from input text
            features = extract_features(text)
            features_array = [features]  # Model expects 2D array
            
            # Make prediction
            prediction = self.model.predict(features_array)[0]
            
            # Get confidence (probability of predicted class)
            # This works if model has predict_proba method (e.g., LogisticRegression, RandomForest)
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_array)[0]
                confidence = max(probabilities)
            else:
                confidence = 1.0  # Fallback for models without predict_proba
            
            return {
                "label": str(prediction),
                "confidence": round(float(confidence), 3)
            }
        except Exception as e:
            print(f"Error making prediction: {e}")
            return {
                "label": "ERROR",
                "confidence": 0.0,
                "error": str(e)
            }


# Global instance
ml_service = MLService()
