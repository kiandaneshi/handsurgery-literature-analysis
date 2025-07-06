import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path

# Import custom modules
from modules.pubmed_retrieval import PubMedRetriever
from modules.biobert_processor import BioBERTProcessor
from modules.data_structuring import DataStructurer
from modules.ml_modeling import MLModeler
from modules.visualization import Visualizer
from config.settings import SEARCH_TERMS, DATE_RANGE

# Page configuration
st.set_page_config(
    page_title="Hand Surgery Literature Analysis Pipeline",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🏥 Hand Surgery Literature Analysis Pipeline")
    st.markdown("### BioBERT-powered Clinical Entity Extraction and ML Prediction")
    
    # Sidebar for navigation
    st.sidebar.title("Pipeline Steps")
    
    # Initialize session state
    if 'data_retrieved' not in st.session_state:
        st.session_state.data_retrieved = False
    if 'nlp_processed' not in st.session_state:
        st.session_state.nlp_processed = False
    if 'data_structured' not in st.session_state:
        st.session_state.data_structured = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    
    # Step selection
    steps = [
        "📚 PubMed Data Retrieval",
        "🧠 BioBERT NLP Processing", 
        "🏗️ Data Structuring",
        "🤖 ML Model Training",
        "📊 Visualization & Results"
    ]
    
    selected_step = st.sidebar.selectbox("Select Pipeline Step", steps)
    
    # Progress indicator
    st.sidebar.markdown("### Progress")
    progress_items = [
        ("Data Retrieved", st.session_state.data_retrieved),
        ("NLP Processed", st.session_state.nlp_processed),
        ("Data Structured", st.session_state.data_structured),
        ("Models Trained", st.session_state.models_trained)
    ]
    
    for item, status in progress_items:
        icon = "✅" if status else "⏳"
        st.sidebar.markdown(f"{icon} {item}")
    
    # Main content based on selected step
    if selected_step == "📚 PubMed Data Retrieval":
        show_pubmed_retrieval()
    elif selected_step == "🧠 BioBERT NLP Processing":
        show_biobert_processing()
    elif selected_step == "🏗️ Data Structuring":
        show_data_structuring()
    elif selected_step == "🤖 ML Model Training":
        show_ml_modeling()
    elif selected_step == "📊 Visualization & Results":
        show_visualization()

def show_pubmed_retrieval():
    st.header("📚 PubMed Literature Retrieval")
    
    # Configuration section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Search Configuration")
        
        # API Key input
        api_key = st.text_input(
            "NCBI API Key (Optional - for 10 RPS vs 3 RPS)",
            value=os.getenv("NCBI_API_KEY", ""),
            type="password",
            help="Get your API key from NCBI account settings for higher rate limits"
        )
        
        # Email for Entrez
        email = st.text_input(
            "Email Address (Required by NCBI)",
            value=os.getenv("ENTREZ_EMAIL", ""),
            help="Required by NCBI for API access"
        )
        
        # Date range
        start_year = st.selectbox("Start Year", range(2013, 2024), index=0)
        end_year = st.selectbox("End Year", range(2013, 2024), index=10)
        
        # Maximum records per search
        max_records = st.number_input(
            "Max Records per Search Term",
            min_value=1000,
            max_value=10000,
            value=5000,
            help="PubMed API limits searches to 10,000 records maximum"
        )
    
    with col2:
        st.subheader("Search Terms")
        st.info("Pre-configured MeSH terms for comprehensive hand surgery literature")
        
        # Display search terms
        for i, term in enumerate(SEARCH_TERMS, 1):
            st.write(f"{i}. {term}")
    
    # Retrieval section
    st.subheader("Data Retrieval")
    
    if not email:
        st.error("Please provide an email address - required by NCBI API")
        return
    
    if st.button("🚀 Start PubMed Retrieval", type="primary"):
        if 'retriever' not in st.session_state:
            st.session_state.retriever = PubMedRetriever(
                email=email,
                api_key=api_key if api_key else None
            )
        
        with st.spinner("Retrieving abstracts from PubMed..."):
            try:
                # Create progress placeholder
                progress_placeholder = st.empty()
                results_placeholder = st.empty()
                
                # Retrieve data
                abstracts_df = st.session_state.retriever.retrieve_abstracts(
                    search_terms=SEARCH_TERMS,
                    start_year=start_year,
                    end_year=end_year,
                    max_records=max_records,
                    progress_callback=lambda msg: progress_placeholder.info(msg)
                )
                
                if abstracts_df is not None and len(abstracts_df) > 0:
                    st.session_state.abstracts_df = abstracts_df
                    st.session_state.data_retrieved = True
                    
                    # Display results
                    st.success(f"✅ Successfully retrieved {len(abstracts_df)} unique abstracts!")
                    
                    # Show summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Abstracts", len(abstracts_df))
                    with col2:
                        st.metric("Unique PMIDs", abstracts_df['pmid'].nunique())
                    with col3:
                        year_range = f"{abstracts_df['year'].min()}-{abstracts_df['year'].max()}"
                        st.metric("Year Range", year_range)
                    with col4:
                        st.metric("Unique Journals", abstracts_df['journal'].nunique())
                    
                    # Show sample data
                    st.subheader("Sample Retrieved Data")
                    st.dataframe(abstracts_df.head(10))
                    
                    # Download option
                    csv = abstracts_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Raw Data (CSV)",
                        data=csv,
                        file_name=f"hand_surgery_abstracts_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.error("No abstracts retrieved. Please check your search parameters.")
                    
            except Exception as e:
                st.error(f"Error during retrieval: {str(e)}")
    
    # Show existing data if available
    if st.session_state.data_retrieved and 'abstracts_df' in st.session_state:
        st.subheader("Current Dataset")
        st.info(f"Dataset contains {len(st.session_state.abstracts_df)} abstracts")
        
        # Show data overview
        if st.checkbox("Show Data Overview"):
            df = st.session_state.abstracts_df
            
            # Year distribution
            st.write("**Publication Year Distribution:**")
            year_counts = df['year'].value_counts().sort_index()
            st.bar_chart(year_counts)
            
            # Journal distribution (top 10)
            st.write("**Top Journals (by article count):**")
            journal_counts = df['journal'].value_counts().head(10)
            st.bar_chart(journal_counts)

def show_biobert_processing():
    st.header("🧠 BioBERT NLP Processing")
    
    if not st.session_state.data_retrieved:
        st.warning("Please complete PubMed data retrieval first.")
        return
    
    st.subheader("Clinical Entity Recognition & Relationship Extraction")
    
    # Model configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Settings")
        
        model_name = st.selectbox(
            "BioBERT Model",
            ["dmis-lab/biobert-base-cased-v1.1", "dmis-lab/biobert-v1.1"],
            help="Pre-trained BioBERT model for clinical NLP"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.1,
            help="Filter out low-confidence extractions"
        )
        
        batch_size = st.number_input(
            "Processing Batch Size",
            min_value=1,
            max_value=32,
            value=8,
            help="Number of abstracts to process in each batch"
        )
    
    with col2:
        st.subheader("Extraction Targets")
        st.info("Entities to extract from each abstract:")
        
        extraction_targets = [
            "**Demographic Data**: age group, gender, population type",
            "**Injury/Procedure**: tendon zone, fracture type, nerve type",
            "**Surgical Interventions**: grafting, fixation, decompression",
            "**Clinical Outcomes**: grip strength, ROM, pain reduction"
        ]
        
        for target in extraction_targets:
            st.write(f"• {target}")
    
    # Processing section
    if st.button("🔬 Start BioBERT Processing", type="primary"):
        if 'biobert_processor' not in st.session_state:
            st.session_state.biobert_processor = BioBERTProcessor(
                model_name=model_name,
                confidence_threshold=confidence_threshold
            )
        
        with st.spinner("Loading BioBERT model and processing abstracts..."):
            try:
                # Initialize processor
                processor = st.session_state.biobert_processor
                
                # Process abstracts
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                processed_data = processor.process_abstracts(
                    st.session_state.abstracts_df,
                    batch_size=batch_size,
                    progress_callback=lambda i, total, msg: (
                        progress_bar.progress(i / total),
                        status_text.text(msg)
                    )
                )
                
                if processed_data is not None:
                    st.session_state.processed_data = processed_data
                    st.session_state.nlp_processed = True
                    
                    st.success("✅ BioBERT processing completed!")
                    
                    # Display results summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Processed Abstracts", len(processed_data))
                    with col2:
                        high_conf = sum(1 for d in processed_data if d['confidence_score'] >= confidence_threshold)
                        st.metric("High Confidence", high_conf)
                    with col3:
                        avg_entities = np.mean([len(d['entities']) for d in processed_data])
                        st.metric("Avg Entities/Abstract", f"{avg_entities:.1f}")
                    with col4:
                        avg_confidence = np.mean([d['confidence_score'] for d in processed_data])
                        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                    
                    # Show sample processed data
                    st.subheader("Sample Processed Results")
                    
                    # Select a high-confidence example
                    high_conf_examples = [d for d in processed_data if d['confidence_score'] >= confidence_threshold]
                    if high_conf_examples:
                        example = high_conf_examples[0]
                        
                        st.write(f"**PMID**: {example['pmid']}")
                        st.write(f"**Title**: {example['title']}")
                        st.write(f"**Confidence Score**: {example['confidence_score']:.2f}")
                        
                        st.write("**Extracted Entities**:")
                        for category, entities in example['entities'].items():
                            if entities:
                                st.write(f"- **{category}**: {', '.join(entities)}")
                        
                        if example.get('relationships'):
                            st.write("**Relationships**:")
                            for rel in example['relationships'][:3]:  # Show first 3
                                st.write(f"- {rel}")
                
                else:
                    st.error("Processing failed. Please check the model configuration.")
                    
            except Exception as e:
                st.error(f"Error during BioBERT processing: {str(e)}")
    
    # Show existing processed data if available
    if st.session_state.nlp_processed and 'processed_data' in st.session_state:
        st.subheader("Processing Results Overview")
        
        data = st.session_state.processed_data
        
        # Confidence distribution
        confidence_scores = [d['confidence_score'] for d in data]
        st.write("**Confidence Score Distribution:**")
        st.histogram(confidence_scores, bins=20)
        
        # Entity extraction statistics
        if st.checkbox("Show Entity Extraction Statistics"):
            entity_stats = {}
            for item in data:
                for category, entities in item['entities'].items():
                    if category not in entity_stats:
                        entity_stats[category] = 0
                    entity_stats[category] += len(entities)
            
            st.write("**Total Entities Extracted by Category:**")
            for category, count in entity_stats.items():
                st.write(f"- {category}: {count}")

def show_data_structuring():
    st.header("🏗️ Structured Data Conversion")
    
    if not st.session_state.nlp_processed:
        st.warning("Please complete BioBERT NLP processing first.")
        return
    
    st.subheader("Converting NLP Outputs to ML-Ready Features")
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Engineering")
        
        # Age group mapping
        st.write("**Age Group Encoding:**")
        st.write("• Pediatric: 0-18 years")
        st.write("• Adult: 19-65 years") 
        st.write("• Elderly: 65+ years")
        
        # Encoding method
        encoding_method = st.selectbox(
            "Categorical Encoding Method",
            ["One-Hot Encoding", "Ordinal Encoding"],
            help="Method for encoding categorical variables"
        )
        
        # Feature selection
        include_text_features = st.checkbox(
            "Include Text-based Features",
            value=True,
            help="Include TF-IDF features from abstract text"
        )
    
    with col2:
        st.subheader("Target Variables")
        st.info("ML model targets to create:")
        
        targets = [
            "**Complication Risk**: Binary classification (yes/no)",
            "**Return to Function**: Regression (days to recovery)",
            "**Surgical Success**: Binary classification (yes/no)"
        ]
        
        for target in targets:
            st.write(f"• {target}")
    
    # Data structuring process
    if st.button("🔧 Structure Data for ML", type="primary"):
        if 'data_structurer' not in st.session_state:
            st.session_state.data_structurer = DataStructurer(
                encoding_method=encoding_method,
                include_text_features=include_text_features
            )
        
        with st.spinner("Converting to structured features..."):
            try:
                structurer = st.session_state.data_structurer
                
                # Structure the data
                structured_data = structurer.structure_data(
                    st.session_state.processed_data,
                    st.session_state.abstracts_df
                )
                
                if structured_data is not None:
                    st.session_state.structured_data = structured_data
                    st.session_state.data_structured = True
                    
                    st.success("✅ Data structuring completed!")
                    
                    # Get feature matrix and targets
                    X, y_dict, feature_names = structured_data
                    
                    # Display summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Samples", X.shape[0])
                    with col2:
                        st.metric("Features", X.shape[1])
                    with col3:
                        st.metric("Target Variables", len(y_dict))
                    with col4:
                        valid_samples = np.sum(~np.isnan(X).any(axis=1))
                        st.metric("Valid Samples", valid_samples)
                    
                    # Feature importance preview
                    st.subheader("Feature Overview")
                    
                    # Show feature categories
                    feature_categories = {}
                    for fname in feature_names:
                        category = fname.split('_')[0] if '_' in fname else 'other'
                        if category not in feature_categories:
                            feature_categories[category] = 0
                        feature_categories[category] += 1
                    
                    st.write("**Features by Category:**")
                    for category, count in feature_categories.items():
                        st.write(f"- {category}: {count} features")
                    
                    # Target variable distributions
                    st.subheader("Target Variable Distributions")
                    
                    for target_name, target_values in y_dict.items():
                        if target_values is not None:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**{target_name}**")
                                if len(np.unique(target_values[~np.isnan(target_values)])) == 2:
                                    # Binary classification
                                    unique_vals, counts = np.unique(target_values[~np.isnan(target_values)], return_counts=True)
                                    st.write(f"Class distribution: {dict(zip(unique_vals, counts))}")
                                else:
                                    # Regression
                                    valid_vals = target_values[~np.isnan(target_values)]
                                    if len(valid_vals) > 0:
                                        st.write(f"Mean: {np.mean(valid_vals):.2f}")
                                        st.write(f"Std: {np.std(valid_vals):.2f}")
                                        st.write(f"Range: {np.min(valid_vals):.2f} - {np.max(valid_vals):.2f}")
                    
                    # Data quality report
                    st.subheader("Data Quality Report")
                    
                    # Missing values
                    missing_features = np.sum(np.isnan(X), axis=0)
                    missing_pct = (missing_features / X.shape[0]) * 100
                    
                    high_missing = np.where(missing_pct > 50)[0]
                    if len(high_missing) > 0:
                        st.warning(f"⚠️ {len(high_missing)} features have >50% missing values")
                    
                    # Feature correlation (sample)
                    if st.checkbox("Show Feature Correlation Analysis"):
                        sample_features = min(20, X.shape[1])
                        corr_matrix = np.corrcoef(X[:, :sample_features].T)
                        
                        import plotly.graph_objects as go
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=corr_matrix,
                            colorscale='RdBu',
                            zmid=0
                        ))
                        fig.update_layout(
                            title=f"Feature Correlation Matrix (First {sample_features} features)",
                            width=600,
                            height=500
                        )
                        st.plotly_chart(fig)
                
                else:
                    st.error("Data structuring failed. Please check the processed data.")
                    
            except Exception as e:
                st.error(f"Error during data structuring: {str(e)}")
    
    # Show existing structured data if available
    if st.session_state.data_structured and 'structured_data' in st.session_state:
        st.subheader("Structured Data Summary")
        
        X, y_dict, feature_names = st.session_state.structured_data
        
        st.info(f"Dataset ready for ML: {X.shape[0]} samples × {X.shape[1]} features")
        
        # Export option
        if st.button("📥 Export Structured Data"):
            # Convert to DataFrame for export
            df_export = pd.DataFrame(X, columns=feature_names)
            
            # Add target variables
            for target_name, target_values in y_dict.items():
                if target_values is not None:
                    df_export[f'target_{target_name}'] = target_values
            
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="Download Structured Dataset (CSV)",
                data=csv,
                file_name=f"structured_hand_surgery_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

def show_ml_modeling():
    st.header("🤖 Machine Learning Model Training")
    
    if not st.session_state.data_structured:
        st.warning("Please complete data structuring first.")
        return
    
    st.subheader("Predictive Model Development")
    
    # Model configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Configuration")
        
        # Train/test split
        test_size = st.slider(
            "Test Set Size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="Proportion of data for testing"
        )
        
        # Cross-validation
        cv_folds = st.selectbox(
            "Cross-Validation Folds",
            [3, 5, 10],
            index=1,
            help="Number of folds for cross-validation"
        )
        
        # Random state
        random_state = st.number_input(
            "Random State",
            value=42,
            help="For reproducible results"
        )
        
        # Model selection
        st.subheader("Models to Train")
        models_to_train = st.multiselect(
            "Select Models",
            ["Logistic Regression", "Random Forest", "XGBoost"],
            default=["Logistic Regression", "Random Forest", "XGBoost"]
        )
    
    with col2:
        st.subheader("Target Tasks")
        
        # Task selection
        tasks_to_train = st.multiselect(
            "Select Prediction Tasks",
            ["Complication Risk", "Return to Function", "Surgical Success"],
            default=["Complication Risk", "Return to Function", "Surgical Success"]
        )
        
        st.info("**Model Types:**")
        st.write("• **Complication Risk**: Binary Classification")
        st.write("• **Return to Function**: Regression")
        st.write("• **Surgical Success**: Binary Classification")
        
        # Performance metrics
        st.subheader("Evaluation Metrics")
        st.write("**Classification**: AUC-ROC, Accuracy, F1-Score")
        st.write("**Regression**: R², RMSE, MAE")
    
    # Model training
    if st.button("🚀 Train Models", type="primary"):
        if not models_to_train:
            st.error("Please select at least one model to train.")
            return
        
        if not tasks_to_train:
            st.error("Please select at least one task to train.")
            return
        
        if 'ml_modeler' not in st.session_state:
            st.session_state.ml_modeler = MLModeler(
                test_size=test_size,
                cv_folds=cv_folds,
                random_state=random_state
            )
        
        with st.spinner("Training machine learning models..."):
            try:
                modeler = st.session_state.ml_modeler
                
                # Train models
                results = modeler.train_models(
                    st.session_state.structured_data,
                    models_to_train=models_to_train,
                    tasks_to_train=tasks_to_train
                )
                
                if results is not None:
                    st.session_state.model_results = results
                    st.session_state.models_trained = True
                    
                    st.success("✅ Model training completed!")
                    
                    # Display results summary
                    st.subheader("Training Results Summary")
                    
                    # Create results table
                    results_data = []
                    
                    for task_name, task_results in results.items():
                        for model_name, model_result in task_results.items():
                            if 'classification' in model_result:
                                # Classification metrics
                                metrics = model_result['classification']
                                results_data.append({
                                    'Task': task_name,
                                    'Model': model_name,
                                    'Type': 'Classification',
                                    'AUC-ROC': f"{metrics['auc_roc']:.3f}",
                                    'Accuracy': f"{metrics['accuracy']:.3f}",
                                    'F1-Score': f"{metrics['f1_score']:.3f}",
                                    'CV Score': f"{metrics['cv_score_mean']:.3f} ± {metrics['cv_score_std']:.3f}"
                                })
                            
                            if 'regression' in model_result:
                                # Regression metrics
                                metrics = model_result['regression']
                                results_data.append({
                                    'Task': task_name,
                                    'Model': model_name,
                                    'Type': 'Regression',
                                    'R²': f"{metrics['r2']:.3f}",
                                    'RMSE': f"{metrics['rmse']:.3f}",
                                    'MAE': f"{metrics['mae']:.3f}",
                                    'CV Score': f"{metrics['cv_score_mean']:.3f} ± {metrics['cv_score_std']:.3f}"
                                })
                    
                    if results_data:
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True)
                    
                    # Best model identification
                    st.subheader("Best Performing Models")
                    
                    for task_name, task_results in results.items():
                        st.write(f"**{task_name}:**")
                        
                        # Find best model for this task
                        best_model = None
                        best_score = -np.inf
                        
                        for model_name, model_result in task_results.items():
                            if 'classification' in model_result:
                                score = model_result['classification']['auc_roc']
                            elif 'regression' in model_result:
                                score = model_result['regression']['r2']
                            else:
                                continue
                            
                            if score > best_score:
                                best_score = score
                                best_model = model_name
                        
                        if best_model:
                            st.write(f"  Best model: {best_model} (Score: {best_score:.3f})")
                    
                    # Feature importance summary
                    st.subheader("Feature Importance Preview")
                    
                    # Show feature importance for one model as example
                    sample_task = list(results.keys())[0]
                    sample_model = list(results[sample_task].keys())[0]
                    
                    if 'feature_importance' in results[sample_task][sample_model]:
                        importance = results[sample_task][sample_model]['feature_importance']
                        
                        # Get top 10 features
                        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                        
                        st.write(f"**Top 10 Features ({sample_task} - {sample_model}):**")
                        for i, (feature, score) in enumerate(top_features, 1):
                            st.write(f"{i}. {feature}: {score:.3f}")
                
                else:
                    st.error("Model training failed. Please check the data and configuration.")
                    
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
    
    # Show existing results if available
    if st.session_state.models_trained and 'model_results' in st.session_state:
        st.subheader("Training Complete")
        st.info("Models are ready for visualization and interpretation.")
        
        # Model export option
        if st.button("💾 Export Model Results"):
            results = st.session_state.model_results
            
            # Convert results to exportable format
            export_data = {}
            for task_name, task_results in results.items():
                export_data[task_name] = {}
                for model_name, model_result in task_results.items():
                    # Extract key metrics
                    if 'classification' in model_result:
                        export_data[task_name][model_name] = model_result['classification']
                    elif 'regression' in model_result:
                        export_data[task_name][model_name] = model_result['regression']
            
            import json
            json_str = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="Download Model Results (JSON)",
                data=json_str,
                file_name=f"model_results_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

def show_visualization():
    st.header("📊 Visualization & Results Interpretation")
    
    if not st.session_state.models_trained:
        st.warning("Please complete model training first.")
        return
    
    st.subheader("Model Performance Analysis & Clinical Interpretation")
    
    # Initialize visualizer
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer()
    
    visualizer = st.session_state.visualizer
    results = st.session_state.model_results
    
    # Visualization options
    viz_options = st.multiselect(
        "Select Visualizations",
        [
            "ROC Curves",
            "Feature Importance", 
            "Predicted vs Actual (Regression)",
            "Performance Comparison",
            "Clinical Interpretation"
        ],
        default=["ROC Curves", "Feature Importance", "Performance Comparison"]
    )
    
    # Generate visualizations
    if st.button("📈 Generate Visualizations", type="primary"):
        with st.spinner("Creating visualizations..."):
            try:
                # ROC Curves
                if "ROC Curves" in viz_options:
                    st.subheader("ROC Curves - Classification Models")
                    
                    roc_figs = visualizer.create_roc_curves(results)
                    for task_name, fig in roc_figs.items():
                        st.write(f"**{task_name}**")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Feature Importance
                if "Feature Importance" in viz_options:
                    st.subheader("Feature Importance Analysis")
                    
                    importance_figs = visualizer.create_feature_importance_plots(results)
                    for (task_name, model_name), fig in importance_figs.items():
                        st.write(f"**{task_name} - {model_name}**")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Predicted vs Actual
                if "Predicted vs Actual (Regression)" in viz_options:
                    st.subheader("Predicted vs Actual - Regression Models")
                    
                    regression_figs = visualizer.create_regression_plots(results)
                    for (task_name, model_name), fig in regression_figs.items():
                        st.write(f"**{task_name} - {model_name}**")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Performance Comparison
                if "Performance Comparison" in viz_options:
                    st.subheader("Model Performance Comparison")
                    
                    comparison_fig = visualizer.create_performance_comparison(results)
                    if comparison_fig:
                        st.plotly_chart(comparison_fig, use_container_width=True)
                
                # Clinical Interpretation
                if "Clinical Interpretation" in viz_options:
                    st.subheader("Clinical Interpretation & Insights")
                    
                    interpretation = visualizer.generate_clinical_interpretation(
                        results,
                        st.session_state.structured_data
                    )
                    
                    # Display interpretation
                    for section, content in interpretation.items():
                        st.write(f"**{section}:**")
                        st.write(content)
                        st.write("")
                
                st.success("✅ Visualizations generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating visualizations: {str(e)}")
    
    # Summary tables
    st.subheader("Summary Tables")
    
    if st.checkbox("Show Detailed Results Tables"):
        
        # Classification results table
        st.write("**Classification Model Results:**")
        classification_data = []
        
        for task_name, task_results in results.items():
            for model_name, model_result in task_results.items():
                if 'classification' in model_result:
                    metrics = model_result['classification']
                    classification_data.append({
                        'Task': task_name,
                        'Model': model_name,
                        'AUC-ROC': metrics['auc_roc'],
                        'Accuracy': metrics['accuracy'],
                        'F1-Score': metrics['f1_score'],
                        'Sensitivity': metrics['sensitivity'],
                        'Specificity': metrics['specificity'],
                        'CV Mean': metrics['cv_score_mean'],
                        'CV Std': metrics['cv_score_std']
                    })
        
        if classification_data:
            class_df = pd.DataFrame(classification_data)
            st.dataframe(class_df.round(3), use_container_width=True)
        
        # Regression results table
        st.write("**Regression Model Results:**")
        regression_data = []
        
        for task_name, task_results in results.items():
            for model_name, model_result in task_results.items():
                if 'regression' in model_result:
                    metrics = model_result['regression']
                    regression_data.append({
                        'Task': task_name,
                        'Model': model_name,
                        'R²': metrics['r2'],
                        'RMSE': metrics['rmse'],
                        'MAE': metrics['mae'],
                        'CV Mean': metrics['cv_score_mean'],
                        'CV Std': metrics['cv_score_std']
                    })
        
        if regression_data:
            reg_df = pd.DataFrame(regression_data)
            st.dataframe(reg_df.round(3), use_container_width=True)
    
    # Export options
    st.subheader("Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export All Visualizations"):
            st.info("Visualization export functionality would save all plots as high-resolution images.")
    
    with col2:
        if st.button("📋 Export Clinical Report"):
            st.info("Clinical report export would generate a comprehensive PDF report.")
    
    with col3:
        if st.button("📈 Export Results Summary"):
            # Create comprehensive results summary
            summary_data = {
                'timestamp': datetime.now().isoformat(),
                'model_results': results,
                'data_summary': {
                    'total_abstracts': len(st.session_state.abstracts_df) if 'abstracts_df' in st.session_state else 0,
                    'processed_abstracts': len(st.session_state.processed_data) if 'processed_data' in st.session_state else 0,
                    'features': st.session_state.structured_data[0].shape[1] if 'structured_data' in st.session_state else 0
                }
            }
            
            import json
            json_str = json.dumps(summary_data, indent=2, default=str)
            
            st.download_button(
                label="Download Complete Results",
                data=json_str,
                file_name=f"hand_surgery_analysis_results_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
