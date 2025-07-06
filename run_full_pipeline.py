#!/usr/bin/env python3
"""
Full execution of the hand surgery literature analysis pipeline.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from pathlib import Path

# Add modules to path
sys.path.append('.')

from modules.pubmed_retrieval import PubMedRetriever
from modules.biobert_processor import BioBERTProcessor
from modules.data_structuring import DataStructurer
from modules.ml_modeling import MLModeler
from modules.visualization import Visualizer
from config.settings import SEARCH_TERMS, DATE_RANGE
from utils.helpers import DataValidator, FileManager, ProgressTracker

def main():
    """Execute the complete hand surgery literature analysis pipeline."""
    
    # Configuration
    email = "kian.daneshi@hotmail.com"
    api_key = "0a9113e6e98389b916269a14820aa5b6c208"
    
    print("🏥 Hand Surgery Literature Analysis Pipeline")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Email: {email}")
    print(f"API Key: {'*' * 20 + api_key[-8:]}")
    print(f"Search Terms: {len(SEARCH_TERMS)} comprehensive MeSH terms")
    print(f"Date Range: {DATE_RANGE['start_year']}-{DATE_RANGE['end_year']}")
    print("=" * 80)
    
    # Create directories for outputs
    Path("results").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    try:
        # Step 1: PubMed Literature Retrieval
        print("\n📚 STEP 1: PubMed Literature Retrieval")
        print("-" * 50)
        
        retriever = PubMedRetriever(email=email, api_key=api_key)
        
        print("Search terms to be used:")
        for i, term in enumerate(SEARCH_TERMS[:10], 1):
            print(f"  {i:2d}. {term}")
        if len(SEARCH_TERMS) > 10:
            print(f"  ... and {len(SEARCH_TERMS) - 10} more terms")
        
        print(f"\nStarting retrieval for {len(SEARCH_TERMS)} search terms...")
        print("This may take 15-30 minutes depending on API rate limits...")
        
        abstracts_df = retriever.retrieve_abstracts(
            search_terms=SEARCH_TERMS,
            start_year=DATE_RANGE['start_year'],
            end_year=DATE_RANGE['end_year'],
            max_records=5000,
            progress_callback=lambda msg: print(f"  → {msg}")
        )
        
        if abstracts_df is None or len(abstracts_df) == 0:
            print("❌ No abstracts retrieved. Pipeline terminated.")
            return
        
        print(f"✅ Successfully retrieved {len(abstracts_df)} unique abstracts")
        
        # Save raw data
        raw_data_file = f"data/hand_surgery_abstracts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        abstracts_df.to_csv(raw_data_file, index=False)
        print(f"📁 Raw data saved to: {raw_data_file}")
        
        # Validation
        validation = DataValidator.validate_pubmed_data(abstracts_df)
        print(f"📊 Data Quality: {len(validation['warnings'])} warnings, {len(validation['errors'])} errors")
        
        # Display statistics
        print("\nDataset Statistics:")
        print(f"  Total abstracts: {len(abstracts_df)}")
        print(f"  Unique PMIDs: {abstracts_df['pmid'].nunique()}")
        print(f"  Year range: {abstracts_df['year'].min()}-{abstracts_df['year'].max()}")
        print(f"  Unique journals: {abstracts_df['journal'].nunique()}")
        print(f"  Average abstract length: {abstracts_df['abstract'].str.len().mean():.0f} characters")
        
        # Step 2: BioBERT NLP Processing
        print("\n\n🧠 STEP 2: BioBERT NLP Processing")
        print("-" * 50)
        
        processor = BioBERTProcessor(
            model_name="dmis-lab/biobert-base-cased-v1.1",
            confidence_threshold=0.3
        )
        
        print("Loading BioBERT model...")
        model_loaded = processor.load_model()
        
        if model_loaded:
            print("✅ BioBERT model loaded successfully")
        else:
            print("⚠️ Using rule-based entity extraction only")
        
        print(f"Processing {len(abstracts_df)} abstracts...")
        print("This may take 30-60 minutes depending on hardware...")
        
        processed_data = processor.process_abstracts(
            abstracts_df,
            batch_size=8,
            progress_callback=lambda i, total, msg: print(f"  → Progress: {i}/{total} - {msg}")
        )
        
        if not processed_data:
            print("❌ BioBERT processing failed. Pipeline terminated.")
            return
        
        print(f"✅ Successfully processed {len(processed_data)} abstracts")
        
        # Save processed data
        processed_data_file = f"data/processed_abstracts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        FileManager.save_data_to_file(processed_data, processed_data_file, 'pickle')
        print(f"📁 Processed data saved to: {processed_data_file}")
        
        # Display processing statistics
        stats = processor.get_processing_statistics(processed_data)
        print("\nProcessing Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Step 3: Data Structuring
        print("\n\n🏗️ STEP 3: Data Structuring")
        print("-" * 50)
        
        structurer = DataStructurer(
            encoding_method="One-Hot Encoding",
            include_text_features=True
        )
        
        print("Converting NLP outputs to ML-ready features...")
        structured_data = structurer.structure_data(processed_data, abstracts_df)
        
        if structured_data is None:
            print("❌ Data structuring failed. Pipeline terminated.")
            return
        
        X, y_dict, feature_names = structured_data
        print(f"✅ Created structured dataset: {X.shape[0]} samples × {X.shape[1]} features")
        
        # Save structured data
        structured_data_file = f"data/structured_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        FileManager.save_data_to_file(structured_data, structured_data_file, 'pickle')
        print(f"📁 Structured data saved to: {structured_data_file}")
        
        # Display feature summary
        feature_summary = structurer.get_feature_summary(X, feature_names, y_dict)
        print("\nFeature Summary:")
        for key, value in feature_summary.items():
            print(f"  {key}: {value}")
        
        # Step 4: Machine Learning Modeling
        print("\n\n🤖 STEP 4: Machine Learning Modeling")
        print("-" * 50)
        
        modeler = MLModeler(test_size=0.2, cv_folds=5, random_state=42)
        
        models_to_train = ['Logistic Regression', 'Random Forest', 'XGBoost']
        tasks_to_train = ['Complication Risk', 'Return to Function', 'Surgical Success']
        
        print(f"Training {len(models_to_train)} models for {len(tasks_to_train)} tasks...")
        print("This may take 10-20 minutes...")
        
        results = modeler.train_models(structured_data, models_to_train, tasks_to_train)
        
        if not results:
            print("❌ Model training failed. Pipeline terminated.")
            return
        
        print("✅ Model training completed successfully")
        
        # Save results
        results_file = f"results/ml_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        FileManager.save_data_to_file(results, results_file, 'pickle')
        print(f"📁 Results saved to: {results_file}")
        
        # Display model performance
        summary_df = modeler.get_model_summary(results)
        print("\nModel Performance Summary:")
        print(summary_df.to_string(index=False))
        
        best_models = modeler.get_best_models(results)
        print("\nBest Performing Models:")
        for task, (model, score) in best_models.items():
            print(f"  {task}: {model} (Score: {score:.3f})")
        
        # Step 5: Visualization and Clinical Interpretation
        print("\n\n📊 STEP 5: Visualization and Clinical Interpretation")
        print("-" * 50)
        
        visualizer = Visualizer()
        
        print("Generating clinical interpretation...")
        interpretation = visualizer.generate_clinical_interpretation(results, structured_data)
        
        # Save interpretation
        interpretation_file = f"results/clinical_interpretation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        FileManager.save_data_to_file(interpretation, interpretation_file, 'json')
        print(f"📁 Clinical interpretation saved to: {interpretation_file}")
        
        print("✅ Visualization and interpretation completed")
        
        # Display key insights
        print("\nKey Clinical Insights:")
        for section, content in interpretation.items():
            print(f"\n{section}:")
            if isinstance(content, str):
                print(f"  {content[:200]}...")
            elif isinstance(content, dict):
                for key, value in list(content.items())[:3]:
                    print(f"  {key}: {value}")
        
        # Final Summary
        print("\n" + "=" * 80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total abstracts processed: {len(abstracts_df)}")
        print(f"High-confidence extractions: {len(processed_data)}")
        print(f"ML features created: {X.shape[1]}")
        print(f"Models trained: {len(models_to_train) * len(tasks_to_train)}")
        print("\nOutput Files:")
        print(f"  📁 Raw data: {raw_data_file}")
        print(f"  📁 Processed data: {processed_data_file}")
        print(f"  📁 Structured data: {structured_data_file}")
        print(f"  📁 ML results: {results_file}")
        print(f"  📁 Clinical interpretation: {interpretation_file}")
        
        return {
            'abstracts_df': abstracts_df,
            'processed_data': processed_data,
            'structured_data': structured_data,
            'results': results,
            'interpretation': interpretation
        }
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    pipeline_results = main()