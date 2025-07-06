"""
Configuration settings for the Hand Surgery Literature Analysis Pipeline.
Contains search terms, model parameters, and application constants.
"""

import os
from typing import List, Dict, Any

# ============================================================================
# PubMed Search Configuration
# ============================================================================

# Comprehensive MeSH terms for hand surgery literature
SEARCH_TERMS = [
    "hand injuries[MeSH Terms]",
    "hand surgery[MeSH Terms]", 
    "tendon injuries[MeSH Terms]",
    "nerve repair[MeSH Terms]",
    "fracture fixation[MeSH Terms]",
    "carpal tunnel syndrome[MeSH Terms]",
    "flexor tendon[MeSH Terms]",
    "nerve grafting[MeSH Terms]",
    "hand transplantation[MeSH Terms]",
    "reconstructive surgical procedures[MeSH Terms]",
    
    # Additional specific terms for comprehensive coverage
    "hand fractures[MeSH Terms]",
    "digital replantation[MeSH Terms]",
    "finger injuries[MeSH Terms]",
    "wrist injuries[MeSH Terms]",
    "metacarpal fractures[MeSH Terms]",
    "phalangeal fractures[MeSH Terms]",
    "scaphoid fractures[MeSH Terms]",
    "radius fractures[MeSH Terms]",
    "ulnar nerve[MeSH Terms]",
    "median nerve[MeSH Terms]",
    "radial nerve[MeSH Terms]",
    "trigger finger[MeSH Terms]",
    "dupuytren contracture[MeSH Terms]",
    "arthrodesis[MeSH Terms] AND hand",
    "arthroplasty[MeSH Terms] AND hand",
    
    # Free text terms for broader coverage
    '"zone II flexor tendon"',
    '"zone I flexor tendon"',
    '"extensor tendon repair"',
    '"hand therapy"',
    '"occupational therapy" AND hand',
    '"grip strength" AND hand',
    '"range of motion" AND hand',
    '"return to work" AND hand surgery',
    '"DASH score" AND hand',
    '"QuickDASH" AND hand'
]

# Date range for literature search
DATE_RANGE = {
    'start_year': 2013,
    'end_year': 2023
}

# PubMed API configuration
PUBMED_CONFIG = {
    'retmax_per_search': 5000,  # Maximum records per search term (PubMed limit: 10,000)
    'batch_size': 200,          # Records to fetch per API call
    'max_total_records': 50000, # Maximum total records to process
    'rate_limit_delay': 0.1,    # Minimum delay between API calls (seconds)
    'timeout': 30,              # API timeout (seconds)
    'max_retries': 3            # Maximum retry attempts for failed requests
}

# ============================================================================
# BioBERT Model Configuration
# ============================================================================

BIOBERT_CONFIG = {
    'model_name': 'dmis-lab/biobert-base-cased-v1.1',
    'alternative_models': [
        'dmis-lab/biobert-v1.1',
        'emilyalsentzer/Bio_ClinicalBERT',
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
    ],
    'confidence_threshold': 0.3,
    'batch_size': 8,
    'max_length': 512,
    'aggregation_strategy': 'simple',
    'device': 'auto'  # Will auto-detect CUDA availability
}

# Clinical entity categories and patterns
ENTITY_CATEGORIES = {
    'demographics': {
        'subcategories': ['age_group', 'gender', 'population_type'],
        'weight': 0.15
    },
    'injury_procedure': {
        'subcategories': ['tendon_zone', 'fracture_type', 'nerve_type', 'anatomical_location'],
        'weight': 0.35
    },
    'surgical_interventions': {
        'subcategories': ['grafting', 'fixation', 'nerve_decompression', 'occupational_therapy', 'splinting'],
        'weight': 0.25
    },
    'clinical_outcomes': {
        'subcategories': ['grip_strength', 'range_of_motion', 'pain_reduction', 'reoperation_rate', 'return_to_work', 'complications'],
        'weight': 0.25
    }
}

# Confidence scoring weights
CONFIDENCE_WEIGHTS = {
    'entity_coverage': 0.3,     # How many entity categories are covered
    'entity_confidence': 0.4,   # Average confidence of BioBERT extractions
    'text_quality': 0.2,        # Text length and structure quality
    'clinical_relevance': 0.1   # Presence of clinical keywords
}

# ============================================================================
# Machine Learning Configuration
# ============================================================================

ML_CONFIG = {
    'test_size': 0.2,
    'cv_folds': 5,
    'random_state': 42,
    'scoring': {
        'classification': ['accuracy', 'f1_weighted', 'roc_auc'],
        'regression': ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    }
}

# Model hyperparameters
MODEL_PARAMS = {
    'LogisticRegression': {
        'max_iter': 1000,
        'class_weight': 'balanced',
        'solver': 'liblinear',
        'random_state': 42
    },
    'RandomForest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'class_weight': 'balanced',
        'random_state': 42
    },
    'XGBoost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'encoding_method': 'one_hot',  # 'one_hot' or 'label'
    'include_text_features': True,
    'tfidf_max_features': 100,
    'tfidf_ngram_range': (1, 2),
    'tfidf_min_df': 2,
    'scale_features': True
}

# ============================================================================
# Clinical Interpretation Settings
# ============================================================================

CLINICAL_THRESHOLDS = {
    'excellent_performance': 0.85,
    'good_performance': 0.75,
    'acceptable_performance': 0.65,
    'high_confidence': 0.7,
    'minimum_confidence': 0.3
}

# Clinical outcome mappings
OUTCOME_MAPPINGS = {
    'age_groups': {
        'pediatric': (0, 18),
        'adult': (19, 65),
        'elderly': (66, 150)
    },
    'tendon_zones': {
        'zone_i': 'Distal to FDS insertion',
        'zone_ii': 'FDS and FDP in fibro-osseous tunnel',
        'zone_iii': 'Proximal to A1 pulley',
        'zone_iv': 'Carpal tunnel',
        'zone_v': 'Proximal to carpal tunnel'
    },
    'complications': {
        'major': ['infection', 'rupture', 'nonunion', 'malunion', 'nerve injury'],
        'minor': ['stiffness', 'pain', 'scarring', 'swelling']
    }
}

# ============================================================================
# Visualization Configuration
# ============================================================================

VISUALIZATION_CONFIG = {
    'color_palette': {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'warning': '#d62728',
        'info': '#9467bd',
        'models': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    },
    'figure_size': {
        'default': (800, 600),
        'wide': (1200, 600),
        'tall': (800, 1000)
    },
    'font_size': {
        'title': 16,
        'axis': 12,
        'legend': 10
    }
}

# ============================================================================
# File and Directory Configuration
# ============================================================================

DIRECTORIES = {
    'data': 'data',
    'models': 'models',
    'results': 'results',
    'figures': 'figures',
    'logs': 'logs',
    'backups': 'backups'
}

FILE_FORMATS = {
    'raw_data': 'csv',
    'processed_data': 'pickle',
    'models': 'pickle',
    'results': 'json',
    'figures': 'png'
}

# ============================================================================
# Application Metadata
# ============================================================================

APP_INFO = {
    'name': 'Hand Surgery Literature Analysis Pipeline',
    'version': '1.0.0',
    'description': 'BioBERT-powered clinical entity extraction and ML prediction for hand surgery outcomes',
    'author': 'Clinical Research Team',
    'license': 'MIT',
    'repository': 'https://github.com/clinical-research/hand-surgery-analysis'
}

# ============================================================================
# Environment Variables and Secrets
# ============================================================================

# Environment variable names for sensitive data
ENV_VARS = {
    'ncbi_api_key': 'NCBI_API_KEY',
    'entrez_email': 'ENTREZ_EMAIL',
    'huggingface_token': 'HUGGINGFACE_TOKEN'
}

# Default values for development (not for production)
DEFAULT_VALUES = {
    'entrez_email': 'researcher@example.com',
    'batch_processing': True,
    'cache_results': True,
    'log_level': 'INFO'
}

# ============================================================================
# Utility Functions for Configuration Access
# ============================================================================

def get_env_var(var_name: str, default: str = None) -> str:
    """
    Get environment variable with fallback to default.
    
    Args:
        var_name: Environment variable name
        default: Default value if variable not found
        
    Returns:
        Environment variable value or default
    """
    return os.getenv(var_name, default)

def get_api_config() -> Dict[str, Any]:
    """
    Get API configuration with environment variables.
    
    Returns:
        Dictionary with API configuration
    """
    return {
        'ncbi_api_key': get_env_var(ENV_VARS['ncbi_api_key']),
        'entrez_email': get_env_var(ENV_VARS['entrez_email'], DEFAULT_VALUES['entrez_email']),
        'huggingface_token': get_env_var(ENV_VARS['huggingface_token']),
        **PUBMED_CONFIG
    }

def get_model_config(model_type: str = 'biobert') -> Dict[str, Any]:
    """
    Get model configuration for specified type.
    
    Args:
        model_type: Type of model ('biobert', 'ml')
        
    Returns:
        Model configuration dictionary
    """
    if model_type == 'biobert':
        return BIOBERT_CONFIG
    elif model_type == 'ml':
        return ML_CONFIG
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_clinical_config() -> Dict[str, Any]:
    """
    Get clinical configuration settings.
    
    Returns:
        Clinical configuration dictionary
    """
    return {
        'entity_categories': ENTITY_CATEGORIES,
        'confidence_weights': CONFIDENCE_WEIGHTS,
        'thresholds': CLINICAL_THRESHOLDS,
        'outcome_mappings': OUTCOME_MAPPINGS
    }

def validate_config() -> Dict[str, bool]:
    """
    Validate configuration settings.
    
    Returns:
        Dictionary with validation results
    """
    validation = {
        'pubmed_config': True,
        'biobert_config': True,
        'ml_config': True,
        'paths': True
    }
    
    # Validate PubMed configuration
    if not SEARCH_TERMS:
        validation['pubmed_config'] = False
    
    if DATE_RANGE['start_year'] >= DATE_RANGE['end_year']:
        validation['pubmed_config'] = False
    
    # Validate BioBERT configuration
    if BIOBERT_CONFIG['confidence_threshold'] < 0 or BIOBERT_CONFIG['confidence_threshold'] > 1:
        validation['biobert_config'] = False
    
    # Validate ML configuration
    if ML_CONFIG['test_size'] <= 0 or ML_CONFIG['test_size'] >= 1:
        validation['ml_config'] = False
    
    return validation

# ============================================================================
# Export key configurations for easy import
# ============================================================================

__all__ = [
    'SEARCH_TERMS',
    'DATE_RANGE', 
    'PUBMED_CONFIG',
    'BIOBERT_CONFIG',
    'ENTITY_CATEGORIES',
    'CONFIDENCE_WEIGHTS',
    'ML_CONFIG',
    'MODEL_PARAMS',
    'FEATURE_CONFIG',
    'CLINICAL_THRESHOLDS',
    'OUTCOME_MAPPINGS',
    'VISUALIZATION_CONFIG',
    'APP_INFO',
    'get_env_var',
    'get_api_config',
    'get_model_config',
    'get_clinical_config',
    'validate_config'
]
