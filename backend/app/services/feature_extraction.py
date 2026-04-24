"""Feature extraction for ML model"""
import re
from typing import List


def clean_text(text: str) -> str:
    """
    Simple text cleaning.
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_features(text: str) -> List[float]:
    """
    Extract simple features from text for ML model.
    
    In a real scenario, you would:
    - Use TF-IDF vectors
    - Use word embeddings
    - Use pretrained feature extractors
    
    Here we use very simple features just for demonstration.
    
    Args:
        text: Input text
        
    Returns:
        Feature vector (list of floats)
    """
    cleaned = clean_text(text)
    words = cleaned.split()
    
    features = [
        len(text),  # Original length
        len(words),  # Word count
        len(set(words)),  # Unique word count
        sum(len(w) for w in words) / max(len(words), 1),  # Average word length
    ]
    
    return features
