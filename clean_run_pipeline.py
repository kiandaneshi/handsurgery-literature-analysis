#!/usr/bin/env python3
"""
Complete execution script for the hand surgery literature analysis pipeline.

This script runs the entire pipeline from PubMed retrieval through machine learning
modeling and clinical interpretation.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from pathlib import Path
import logging
import json
import pickle

# Add src to path
sys.path.append('src')

# Import pipeline modules
from pubmed_retrieval import PubMedRetriever
from biobert_processor import BioBERTProcessor
from data_structuring import DataStructurer
from ml_modeling import MLModeler
from visualization import Visualizer

# Import configuration
import clean_config as config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for the pipeline."""
    directories = ['data', 'results', 'figures', 'models', 'logs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Directory '{directory}' ready")

def run_pubmed_retrieval(
    email: str,
    api_key: str = None,
    max_records: int = 1000,
    start_year: int = 2013,
    end_year: int = 2023
) -> pd.DataFrame:
    """
    Execute PubMed literature retrieval.
    
    Args:
        email: User email for NCBI API
        api_key: Optional NCBI API key
        max_records: Maximum records per search term
        start_year: Start year for search
        end_year: End year for search
        
    Returns:
        DataFrame with retrieved abstracts or None if failed
    """
    logger.info("=" * 80)
    logger.info("STEP 1: PubMed Literature Retrieval")
    logger.info("=" * 80)
    
    try:
        # Initialize retriever
        retriever = PubMedRetriever(email=email, api_key=api_key)
        
        # Get search terms from config
        search_terms = config.SEARCH_TERMS
        
        logger.info(f"Configuration:")
        logger.info(f"  - Email: {email}")
        logger.info(f"  - API Key: {'Provided' if api_key else 'Not provided'}")
        logger.info(f"  - Search terms: {len(search_terms)}")
        logger.info(f"  - Date range: {start_year}-{end_year}")
        logger.info(f"  - Max records per term: {max_records}")
        
        def progress_callback(message):
            logger.info(f"  → {message}")
        
        # Execute retrieval
        abstracts_df = retriever.retrieve_abstracts(
            search_terms=search_terms,
            start_year=start_year,
            end_year=end_year,
            max_records=max_records,
            progress_callback=progress_callback
        )
        
        if abstracts_df is not None and len(abstracts_df) > 0:
            logger.info(f"✅ Successfully retrieved {len(abstracts_df)} abstracts")
            
            # Save raw data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/raw_abstracts_{timestamp}.csv"
            abstracts_df.to_csv(filename, index=False)
            logger.info(f"💾 Raw data saved to {filename}")
            
            # Display statistics
            stats = retriever.get_retrieval_statistics(abstracts_df)
            logger.info(f"📊 Dataset Statistics:")
            logger.info(f"  - Total abstracts: {stats.get('total_abstracts', 0):,}")
            logger.info(f"  - Unique PMIDs: {stats.get('unique_pmids', 0):,}")
            logger.info(f"  - Unique journals: {stats.get('unique_journals', 0):,}")
            logger.info(f"  - Year range: {stats.get('year_range', 'N/A')}")
            
            return abstracts_df
        else:
            logger.error("❌ No abstracts retrieved")
            return None
            
    except Exception as e:
        logger.error(f"❌ PubMed retrieval failed: {str(e)}")
        return None

def run_biobert_processing(
    abstracts_df: pd.DataFrame,
    confidence_threshold: float = 0.3,
    batch_size: int = 8
) -> list:
    """
    Execute BioBERT NLP processing.
    
    Args:
        abstracts_df: DataFrame with abstracts
        confidence_threshold: Minimum confidence for extractions
        batch_size: Batch size for processing
        
    Returns:
        List of processed abstracts or None if failed
    """
    logger.info("=" * 80)
    logger.info("STEP 2: BioBERT NLP Processing")
    logger.info("=" * 80)
    
    try:
        # Initialize processor
        processor = BioBERTProcessor(
            model_name=config.BIOBERT_CONFIG['model_name'],
            confidence_threshold=confidence_threshold
        )
        
        logger.info(f"Configuration:")
        logger.info(f"  - Model: {config.BIOBERT_CONFIG['model_name']}")
        logger.info(f"  - Confidence threshold: {confidence_threshold}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Abstracts to process: {len(abstracts_df):,}")
        
        # Load model
        logger.info("Loading BioBERT model...")
        model_loaded = processor.load_model()
        
        if model_loaded:
            logger.info("✅ BioBERT model loaded successfully")
        else:
            logger.warning("⚠️ Using rule-based extraction only")
        
        def progress_callback(current, total, message):
            if current % 50 == 0:  # Log every 50 abstracts
                progress_pct = (current / total) * 100
                logger.info(f"  → Progress: {current:,}/{total:,} ({progress_pct:.1f}%) - {message}")
        
        # Execute processing
        processed_data = processor.process_abstracts(
            abstracts_df,
            batch_size=batch_size,
            progress_callback=progress_callback
        )
        
        if processed_data:
            logger.info(f"✅ Successfully processed {len(processed_data):,} abstracts")
            
            # Generate statistics
            stats = processor.get_processing_statistics(processed_data)
            logger.info(f"📊 Processing Statistics:")
            logger.info(f"  - Success rate: {stats.get('processing_success_rate', 0):.1%}")
            logger.info(f"  - Average confidence: {stats.get('avg_confidence', 0):.3f}")
            logger.info(f"  - Total entities: {stats.get('total_entities_extracted', 0):,}")
            logger.info(f"  - Avg entities per abstract: {stats.get('avg_entities_per_abstract', 0):.1f}")
            
            # Save processed data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/processed_abstracts_{timestamp}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(processed_data, f)
            logger.info(f"💾 Processed data saved to {filename}")
            
            return processed_data
        else:
            logger.error("❌ BioBERT processing failed")
            return None
            
    except Exception as e:
        logger.error(f"❌ BioBERT processing failed: {str(e)}")
        return None

def run_data_structuring(
    processed_data: list,
    abstracts_df: pd.DataFrame
) -> tuple:
    """
    Execute data structuring and feature engineering.
    
    Args:
        processed_data: List of processed abstracts
        abstracts_df: Original abstracts DataFrame
        
    Returns:
        Structured data tuple or None if failed
    """
    logger.info("=" * 80)
    logger.info("STEP 3: Data Structuring & Feature Engineering")
    logger.info("=" * 80)
    
    try:
        # Initialize structurer
        structurer = DataStructurer(
            encoding_method=config.FEATURE_CONFIG['encoding_method'],
            include_text_features=config.FEATURE_CONFIG['include_text_features']
        )
        
        logger.info(f"Configuration:")
        logger.info(f"  - Encoding method: {config.FEATURE_CONFIG['encoding_method']}")
        logger.info(f"  - Include text features: {config.FEATURE_CONFIG['include_text_features']}")
        logger.info(f"  - Processed abstracts: {len(processed_data):,}")
        
        # Execute structuring
        structured_data = structurer.structure_data(processed_data, abstracts_df)
        
        if structured_data is not None:
            X, y_dict, feature_names = structured_data
            
            logger.info(f"✅ Data structuring completed")
            logger.info(f"📊 Structured Data Statistics:")
            logger.info(f"  - Samples: {X.shape[0]:,}")
            logger.info(f"  - Features: {X.shape[1]:,}")
            logger.info(f"  - Target tasks: {len(y_dict)}")
            
            # Feature summary
            summary = structurer.get_feature_summary(X, feature_names, y_dict)
            logger.info(f"  - Feature density: {summary.get('data_quality', {}).get('feature_density', 0):.1%}")
            
            # Save structured data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/structured_data_{timestamp}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(structured_data, f)
            logger.info(f"💾 Structured data saved to {filename}")
            
            return structured_data
        else:
            logger.error("❌ Data structuring failed")
            return None
            
    except Exception as e:
        logger.error(f"❌ Data structuring failed: {str(e)}")
        return None

def run_ml_modeling(
    structured_data: tuple,
    models_to_train: list = None
) -> dict:
    """
    Execute machine learning modeling.
    
    Args:
        structured_data: Structured data tuple
        models_to_train: List of model names to train
        
    Returns:
        ML results dictionary or None if failed
    """
    logger.info("=" * 80)
    logger.info("STEP 4: Machine Learning Modeling")
    logger.info("=" * 80)
    
    try:
        X, y_dict, feature_names = structured_data
        
        if models_to_train is None:
            models_to_train = config.ML_CONFIG['models']
        
        # Initialize modeler
        modeler = MLModeler(
            test_size=config.ML_CONFIG['test_size'],
            cv_folds=config.ML_CONFIG['cv_folds'],
            random_state=config.ML_CONFIG['random_state']
        )
        
        logger.info(f"Configuration:")
        logger.info(f"  - Models: {', '.join(models_to_train)}")
        logger.info(f"  - Test size: {config.ML_CONFIG['test_size']}")
        logger.info(f"  - CV folds: {config.ML_CONFIG['cv_folds']}")
        logger.info(f"  - Dataset shape: {X.shape}")
        
        # Filter tasks with sufficient data
        valid_tasks = []
        for task_name, y_values in y_dict.items():
            valid_count = np.sum(~np.isnan(y_values))
            if valid_count >= 50:  # Minimum samples required
                valid_tasks.append(task_name)
                logger.info(f"  - Task '{task_name}': {valid_count:,} valid samples")
            else:
                logger.warning(f"  - Task '{task_name}': Only {valid_count} samples (skipped)")
        
        if not valid_tasks:
            logger.error("❌ No tasks have sufficient data for modeling")
            return None
        
        # Execute modeling
        logger.info(f"Training {len(models_to_train)} models on {len(valid_tasks)} tasks...")
        
        results = modeler.train_models(structured_data, models_to_train, valid_tasks)
        
        if results:
            logger.info(f"✅ Model training completed")
            
            # Display results summary
            summary_df = modeler.get_model_summary(results)
            logger.info(f"📊 Model Performance Summary:")
            logger.info(f"\n{summary_df.to_string(index=False)}")
            
            # Best models
            best_models = modeler.get_best_models(results)
            logger.info(f"🏆 Best Performing Models:")
            for task, (model, score) in best_models.items():
                logger.info(f"  - {task}: {model} (Score: {score:.3f})")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/ml_results_{timestamp}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
            logger.info(f"💾 Results saved to {filename}")
            
            return results
        else:
            logger.error("❌ Model training failed")
            return None
            
    except Exception as e:
        logger.error(f"❌ ML modeling failed: {str(e)}")
        return None

def run_visualization(
    results: dict,
    structured_data: tuple
) -> dict:
    """
    Execute visualization and clinical interpretation.
    
    Args:
        results: ML results dictionary
        structured_data: Structured data tuple
        
    Returns:
        Clinical interpretation dictionary or None if failed
    """
    logger.info("=" * 80)
    logger.info("STEP 5: Visualization & Clinical Interpretation")
    logger.info("=" * 80)
    
    try:
        # Initialize visualizer
        visualizer = Visualizer()
        
        logger.info("Generating clinical interpretation...")
        
        # Generate interpretation
        interpretation = visualizer.generate_clinical_interpretation(results, structured_data)
        
        if interpretation:
            logger.info(f"✅ Clinical interpretation generated")
            
            # Display key insights
            logger.info(f"📋 Key Clinical Insights:")
            logger.info(f"  - Executive Summary: {len(interpretation.get('executive_summary', ''))} characters")
            logger.info(f"  - Clinical Recommendations: {len(interpretation.get('clinical_recommendations', []))} items")
            logger.info(f"  - Study Limitations: {len(interpretation.get('study_limitations', []))} items")
            
            # Save interpretation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/clinical_interpretation_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(interpretation, f, indent=2)
            logger.info(f"💾 Clinical interpretation saved to {filename}")
            
            return interpretation
        else:
            logger.error("❌ Visualization failed")
            return None
            
    except Exception as e:
        logger.error(f"❌ Visualization failed: {str(e)}")
        return None

def main():
    """Main pipeline execution function."""
    print("🏥 Hand Surgery Literature Analysis Pipeline")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Setup directories
    setup_directories()
    
    # Configuration
    email = os.getenv("NCBI_EMAIL", "researcher@example.com")
    api_key = os.getenv("NCBI_API_KEY", None)
    
    if email == "researcher@example.com":
        logger.warning("⚠️ Please set NCBI_EMAIL environment variable")
        logger.info("Usage: export NCBI_EMAIL='your.email@example.com'")
        logger.info("       export NCBI_API_KEY='your_api_key'  # Optional")
        return
    
    logger.info(f"Configuration:")
    logger.info(f"  - Email: {email}")
    logger.info(f"  - API Key: {'Provided' if api_key else 'Not provided'}")
    
    start_time = datetime.now()
    
    try:
        # Step 1: PubMed Retrieval
        abstracts_df = run_pubmed_retrieval(
            email=email,
            api_key=api_key,
            max_records=1000,
            start_year=config.DATE_RANGE['start_year'],
            end_year=config.DATE_RANGE['end_year']
        )
        
        if abstracts_df is None or len(abstracts_df) == 0:
            logger.error("❌ Pipeline terminated: No abstracts retrieved")
            return
        
        # Step 2: BioBERT Processing
        processed_data = run_biobert_processing(
            abstracts_df,
            confidence_threshold=config.BIOBERT_CONFIG['confidence_threshold'],
            batch_size=config.BIOBERT_CONFIG['batch_size']
        )
        
        if not processed_data:
            logger.error("❌ Pipeline terminated: BioBERT processing failed")
            return
        
        # Step 3: Data Structuring
        structured_data = run_data_structuring(processed_data, abstracts_df)
        
        if structured_data is None:
            logger.error("❌ Pipeline terminated: Data structuring failed")
            return
        
        # Step 4: ML Modeling
        results = run_ml_modeling(structured_data)
        
        if not results:
            logger.error("❌ Pipeline terminated: ML modeling failed")
            return
        
        # Step 5: Visualization
        interpretation = run_visualization(results, structured_data)
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 80)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Duration: {duration}")
        logger.info(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Results saved in: data/, results/, figures/")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()