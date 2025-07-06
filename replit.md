# Hand Surgery Literature Analysis Pipeline

## Overview

This is a comprehensive Streamlit-based clinical research application that automates the analysis of hand surgery literature using BioBERT for natural language processing and machine learning for predictive modeling. The pipeline retrieves medical abstracts from PubMed, processes them for clinical entity extraction, structures the data for analysis, and trains ML models to predict surgical outcomes.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with multi-step pipeline interface
- **Layout**: Wide layout with expandable sidebar navigation
- **State Management**: Session state for tracking pipeline progress across steps
- **Visualization**: Interactive plots using Plotly and Seaborn for model performance visualization

### Backend Architecture
- **Modular Design**: Six main modules handling different pipeline stages
- **Processing Flow**: Sequential pipeline from data retrieval → NLP processing → structuring → ML modeling → visualization
- **API Integration**: Bio.Entrez for PubMed data retrieval with rate limiting
- **ML Framework**: Scikit-learn and XGBoost for predictive modeling

### Data Processing Pipeline
1. **PubMed Retrieval**: Automated literature search using MeSH terms
2. **BioBERT Processing**: Clinical entity recognition and relationship extraction
3. **Data Structuring**: Feature engineering and encoding for ML compatibility
4. **ML Modeling**: Multi-task prediction (complications, return-to-function, surgical success)
5. **Visualization**: Interactive dashboards for model performance and clinical insights

## Key Components

### Data Retrieval (`modules/pubmed_retrieval.py`)
- **Purpose**: Fetches hand surgery literature from PubMed using comprehensive MeSH terms
- **Features**: Rate limiting compliance, batch processing, XML parsing
- **API**: Bio.Entrez with optional API key support (10 RPS vs 3 RPS)

### NLP Processing (`modules/biobert_processor.py`)
- **Model**: dmis-lab/biobert-base-cased-v1.1 (Hugging Face)
- **Extraction**: Demographics, injury characteristics, surgical interventions, clinical outcomes
- **Enhancement**: Rule-based patterns for ambiguous BioBERT predictions
- **Confidence Scoring**: Multi-factor scoring based on entity coverage and text quality

### Data Structuring (`modules/data_structuring.py`)
- **Feature Engineering**: Age group mapping, intervention encoding, outcome categorization
- **Encoding Methods**: One-hot encoding, label encoding, TF-IDF vectorization
- **Preprocessing**: StandardScaler normalization and missing value imputation

### ML Modeling (`modules/ml_modeling.py`)
- **Classification Tasks**: Complication risk, surgical outcome success (binary)
- **Regression Task**: Return-to-function time prediction
- **Algorithms**: Logistic Regression, Random Forest, XGBoost
- **Evaluation**: 5-fold cross-validation, AUC-ROC, F1-score, sensitivity, specificity

### Visualization (`modules/visualization.py`)
- **Interactive Plots**: ROC curves, feature importance, performance metrics
- **Clinical Dashboards**: Model comparison, outcome prediction visualization
- **Export Capabilities**: Results export for clinical reporting

## Data Flow

1. **Search Configuration**: MeSH terms and date ranges defined in `config/settings.py`
2. **Literature Retrieval**: PubMed API calls with rate limiting and batch processing
3. **Text Processing**: BioBERT entity extraction with confidence scoring
4. **Feature Engineering**: Conversion to structured ML-ready features
5. **Model Training**: Multi-task learning with cross-validation
6. **Results Visualization**: Interactive dashboards for clinical interpretation

## External Dependencies

### Core ML/NLP Libraries
- **Transformers**: Hugging Face BioBERT model integration
- **PyTorch**: Deep learning backend for BioBERT
- **Scikit-learn**: Classical ML algorithms and preprocessing
- **XGBoost**: Gradient boosting for advanced predictions

### Data Processing
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Biopython**: PubMed API access via Bio.Entrez
- **Streamlit**: Web application framework

### Visualization
- **Plotly**: Interactive plotting and dashboards
- **Seaborn/Matplotlib**: Statistical visualizations

### API Services
- **NCBI Entrez**: PubMed literature database access
- **Hugging Face Hub**: Pre-trained BioBERT model hosting

## Deployment Strategy

### Development Environment
- **Platform**: Replit-compatible Python environment
- **Dependencies**: Requirements managed through pip/conda
- **Data Storage**: Local file system for cached results and models

### Production Considerations
- **Scalability**: Batch processing with configurable rate limits
- **Caching**: Results caching to avoid redundant API calls
- **Error Handling**: Comprehensive validation and error recovery
- **Memory Management**: Efficient processing for large literature datasets

### Configuration Management
- **Settings**: Centralized configuration in `config/settings.py`
- **Search Terms**: Comprehensive MeSH term coverage for hand surgery
- **Model Parameters**: Configurable thresholds and hyperparameters

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes

- July 06, 2025: COMPLETE PIPELINE SUCCESS - Processed 43,726 abstracts with 99.1% success rate
- July 06, 2025: BioBERT extracted entities from 43,249 abstracts with 0.537 avg confidence
- July 06, 2025: Created ML dataset with 43,249 samples × 71 features  
- July 06, 2025: Trained 9 models achieving 98.0% accuracy (complication risk) and 89.7% (surgical success)
- July 06, 2025: Generated comprehensive clinical interpretation and practice recommendations

## Changelog

- July 06, 2025: Complete hand surgery literature analysis pipeline deployed
- July 06, 2025: BioBERT processor with fallback to rule-based extraction
- July 06, 2025: Full ML pipeline with 3 models and 3 prediction tasks
- July 06, 2025: Real-time execution with user's NCBI API credentials