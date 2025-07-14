"""
Configuration settings for Hand Surgery Literature Analysis Pipeline
"""

from typing import Dict, List, Any

# Search terms for comprehensive hand surgery literature coverage
SEARCH_TERMS = [
    # Core hand surgery terms
    "hand surgery[MeSH Terms]",
    "hand injuries[MeSH Terms]",
    "finger injuries[MeSH Terms]",
    "wrist injuries[MeSH Terms]",
    
    # Specific fracture types
    "metacarpal fractures[MeSH Terms]",
    "phalanx fractures[MeSH Terms]",
    "carpal fractures[MeSH Terms]",
    "scaphoid fractures[MeSH Terms]",
    
    # Soft tissue injuries
    "tendon injuries[MeSH Terms] AND hand",
    "tendon repair[MeSH Terms] AND hand",
    "nerve injuries[MeSH Terms] AND hand",
    "nerve repair[MeSH Terms] AND hand",
    "ligament injuries[MeSH Terms] AND hand",
    
    # Surgical procedures
    "microsurgery[MeSH Terms] AND hand",
    "reconstructive surgical procedures[MeSH Terms] AND hand",
    "internal fixation[MeSH Terms] AND hand",
    "external fixation[MeSH Terms] AND hand",
    "bone grafting[MeSH Terms] AND hand",
    "free tissue flaps[MeSH Terms] AND hand",
    
    # Conditions and diseases
    "dupuytren contracture[MeSH Terms]",
    "carpal tunnel syndrome[MeSH Terms]",
    "trigger finger[MeSH Terms]",
    "arthritis[MeSH Terms] AND hand",
    "osteoarthritis[MeSH Terms] AND hand",
    "rheumatoid arthritis[MeSH Terms] AND hand",
    
    # Complications and outcomes
    "postoperative complications[MeSH Terms] AND hand surgery",
    "wound healing[MeSH Terms] AND hand",
    "infection[MeSH Terms] AND hand surgery",
    "range of motion[MeSH Terms] AND hand",
    "functional outcomes[MeSH Terms] AND hand",
    "patient satisfaction[MeSH Terms] AND hand surgery",
    "return to work[MeSH Terms] AND hand surgery",
    
    # Pediatric hand surgery
    "pediatric[MeSH Terms] AND hand surgery",
    "congenital hand deformities[MeSH Terms]",
    "polydactyly[MeSH Terms]",
    "syndactyly[MeSH Terms]"
]

# Date range for literature search
DATE_RANGE = {
    'start_year': 2013,
    'end_year': 2023
}

# BioBERT configuration
BIOBERT_CONFIG = {
    'model_name': 'dmis-lab/biobert-base-cased-v1.1',
    'confidence_threshold': 0.3,
    'max_length': 512,
    'batch_size': 8
}

# Machine learning configuration
ML_CONFIG = {
    'test_size': 0.2,
    'cv_folds': 5,
    'random_state': 42,
    'models': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'scoring_metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'encoding_method': 'One-Hot Encoding',
    'include_text_features': True,
    'text_vectorizer': 'TF-IDF',
    'max_features': 500,
    'min_df': 2,
    'max_df': 0.95
}

# Data quality thresholds
QUALITY_THRESHOLDS = {
    'min_abstract_length': 50,
    'max_abstract_length': 10000,
    'min_year': 1990,
    'max_year': 2024,
    'min_confidence_score': 0.1
}

# Output directories
OUTPUT_DIRS = {
    'data': 'data',
    'results': 'results',
    'figures': 'figures',
    'models': 'models'
}

def get_config() -> Dict[str, Any]:
    """
    Get complete configuration dictionary.
    
    Returns:
        Dictionary containing all configuration settings
    """
    return {
        'search_terms': SEARCH_TERMS,
        'date_range': DATE_RANGE,
        'biobert': BIOBERT_CONFIG,
        'ml': ML_CONFIG,
        'features': FEATURE_CONFIG,
        'quality': QUALITY_THRESHOLDS,
        'output': OUTPUT_DIRS
    }

def validate_config() -> bool:
    """
    Validate configuration settings.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        # Check date range
        if DATE_RANGE['start_year'] >= DATE_RANGE['end_year']:
            return False
        
        # Check quality thresholds
        if QUALITY_THRESHOLDS['min_abstract_length'] <= 0:
            return False
        
        # Check ML configuration
        if ML_CONFIG['test_size'] <= 0 or ML_CONFIG['test_size'] >= 1:
            return False
        
        if ML_CONFIG['cv_folds'] < 2:
            return False
        
        return True
        
    except Exception:
        return False