import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

class Visualizer:
    """
    Handles visualization and interpretation of machine learning results for clinical analysis.
    Creates interactive plots and generates clinical insights from model performance.
    """
    
    def __init__(self):
        """Initialize the visualizer with plotting configurations."""
        # Color schemes for consistency
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'models': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        }
        
        # Plot configurations
        self.default_layout = {
            'font': {'size': 12},
            'showlegend': True,
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white'
        }
    
    def create_roc_curves(self, results: Dict) -> Dict[str, go.Figure]:
        """
        Create ROC curves for classification models.
        
        Args:
            results: Dictionary with model results
            
        Returns:
            Dictionary mapping task names to ROC curve figures
        """
        roc_figures = {}
        
        for task_name, task_results in results.items():
            # Check if this is a classification task
            classification_models = {}
            
            for model_name, model_result in task_results.items():
                if 'classification' in model_result:
                    classification_result = model_result['classification']
                    if 'roc_curve' in classification_result:
                        classification_models[model_name] = classification_result
            
            if not classification_models:
                continue
            
            # Create ROC curve plot
            fig = go.Figure()
            
            for i, (model_name, model_data) in enumerate(classification_models.items()):
                roc_data = model_data['roc_curve']
                auc_score = model_data.get('auc_roc', 0)
                
                fig.add_trace(go.Scatter(
                    x=roc_data['fpr'],
                    y=roc_data['tpr'],
                    mode='lines',
                    name=f"{model_name} (AUC = {auc_score:.3f})",
                    line=dict(color=self.color_palette['models'][i % len(self.color_palette['models'])], width=2)
                ))
            
            # Add diagonal reference line
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='gray', width=1, dash='dash')
            ))
            
            # Update layout
            fig.update_layout(
                title=f'ROC Curves - {task_name}',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                **self.default_layout
            )
            
            roc_figures[task_name] = fig
        
        return roc_figures
    
    def create_feature_importance_plots(self, results: Dict, top_n: int = 20) -> Dict[Tuple[str, str], go.Figure]:
        """
        Create feature importance plots for each model.
        
        Args:
            results: Dictionary with model results
            top_n: Number of top features to show
            
        Returns:
            Dictionary mapping (task_name, model_name) to feature importance figures
        """
        importance_figures = {}
        
        for task_name, task_results in results.items():
            for model_name, model_result in task_results.items():
                
                # Get feature importance from either classification or regression
                feature_importance = None
                if 'classification' in model_result:
                    feature_importance = model_result['classification'].get('feature_importance')
                elif 'regression' in model_result:
                    feature_importance = model_result['regression'].get('feature_importance')
                
                if not feature_importance:
                    continue
                
                # Sort features by importance
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                top_features = sorted_features[:top_n]
                
                if not top_features:
                    continue
                
                # Extract feature names and importance values
                feature_names = [item[0] for item in top_features]
                importance_values = [item[1] for item in top_features]
                
                # Create horizontal bar plot
                fig = go.Figure(go.Bar(
                    x=importance_values,
                    y=feature_names,
                    orientation='h',
                    marker_color=self.color_palette['primary']
                ))
                
                # Update layout
                fig.update_layout(
                    title=f'Feature Importance - {task_name} ({model_name})',
                    xaxis_title='Importance Score',
                    yaxis_title='Features',
                    height=max(400, len(top_features) * 25),
                    yaxis=dict(autorange="reversed"),
                    **self.default_layout
                )
                
                importance_figures[(task_name, model_name)] = fig
        
        return importance_figures
    
    def create_regression_plots(self, results: Dict) -> Dict[Tuple[str, str], go.Figure]:
        """
        Create predicted vs actual plots for regression models.
        
        Args:
            results: Dictionary with model results
            
        Returns:
            Dictionary mapping (task_name, model_name) to regression plots
        """
        regression_figures = {}
        
        for task_name, task_results in results.items():
            for model_name, model_result in task_results.items():
                
                if 'regression' not in model_result:
                    continue
                
                regression_data = model_result['regression']
                
                if 'y_test' not in regression_data or 'y_pred' not in regression_data:
                    continue
                
                y_test = regression_data['y_test']
                y_pred = regression_data['y_pred']
                r2_score = regression_data.get('r2', 0)
                rmse = regression_data.get('rmse', 0)
                
                # Create scatter plot
                fig = go.Figure()
                
                # Add scatter points
                fig.add_trace(go.Scatter(
                    x=y_test,
                    y=y_pred,
                    mode='markers',
                    name='Predictions',
                    marker=dict(
                        color=self.color_palette['primary'],
                        size=8,
                        opacity=0.6
                    )
                ))
                
                # Add perfect prediction line
                min_val = min(np.min(y_test), np.min(y_pred))
                max_val = max(np.max(y_test), np.max(y_pred))
                
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                # Update layout
                fig.update_layout(
                    title=f'Predicted vs Actual - {task_name} ({model_name})<br>R² = {r2_score:.3f}, RMSE = {rmse:.3f}',
                    xaxis_title='Actual Values',
                    yaxis_title='Predicted Values',
                    **self.default_layout
                )
                
                regression_figures[(task_name, model_name)] = fig
        
        return regression_figures
    
    def create_performance_comparison(self, results: Dict) -> Optional[go.Figure]:
        """
        Create a comprehensive performance comparison across all models and tasks.
        
        Args:
            results: Dictionary with model results
            
        Returns:
            Plotly figure with performance comparison or None if no data
        """
        comparison_data = []
        
        for task_name, task_results in results.items():
            for model_name, model_result in task_results.items():
                
                if 'classification' in model_result:
                    metrics = model_result['classification']
                    comparison_data.append({
                        'Task': task_name,
                        'Model': model_name,
                        'Type': 'Classification',
                        'Primary_Score': metrics.get('auc_roc', metrics.get('f1_score', 0)),
                        'Metric_Name': 'AUC-ROC' if 'auc_roc' in metrics else 'F1-Score',
                        'Accuracy': metrics.get('accuracy', 0),
                        'F1_Score': metrics.get('f1_score', 0),
                        'CV_Mean': metrics.get('cv_score_mean', 0)
                    })
                
                elif 'regression' in model_result:
                    metrics = model_result['regression']
                    comparison_data.append({
                        'Task': task_name,
                        'Model': model_name,
                        'Type': 'Regression',
                        'Primary_Score': metrics.get('r2', 0),
                        'Metric_Name': 'R²',
                        'RMSE': metrics.get('rmse', 0),
                        'MAE': metrics.get('mae', 0),
                        'CV_Mean': metrics.get('cv_score_mean', 0)
                    })
        
        if not comparison_data:
            return None
        
        df = pd.DataFrame(comparison_data)
        
        # Create subplots for different metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Primary Performance Metric', 'Cross-Validation Scores', 
                          'Classification Accuracy', 'Regression R²'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Primary performance metric
        for i, task in enumerate(df['Task'].unique()):
            task_data = df[df['Task'] == task]
            fig.add_trace(
                go.Bar(
                    name=task,
                    x=task_data['Model'],
                    y=task_data['Primary_Score'],
                    text=task_data['Metric_Name'],
                    textposition='outside',
                    marker_color=self.color_palette['models'][i % len(self.color_palette['models'])]
                ),
                row=1, col=1
            )
        
        # Cross-validation scores
        for i, task in enumerate(df['Task'].unique()):
            task_data = df[df['Task'] == task]
            fig.add_trace(
                go.Bar(
                    name=f"{task} (CV)",
                    x=task_data['Model'],
                    y=task_data['CV_Mean'],
                    marker_color=self.color_palette['models'][i % len(self.color_palette['models'])],
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Classification accuracy
        classification_data = df[df['Type'] == 'Classification']
        if not classification_data.empty:
            for i, task in enumerate(classification_data['Task'].unique()):
                task_data = classification_data[classification_data['Task'] == task]
                fig.add_trace(
                    go.Bar(
                        name=f"{task} (Accuracy)",
                        x=task_data['Model'],
                        y=task_data['Accuracy'],
                        marker_color=self.color_palette['models'][i % len(self.color_palette['models'])],
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # Regression R²
        regression_data = df[df['Type'] == 'Regression']
        if not regression_data.empty:
            for i, task in enumerate(regression_data['Task'].unique()):
                task_data = regression_data[regression_data['Task'] == task]
                fig.add_trace(
                    go.Bar(
                        name=f"{task} (R²)",
                        x=task_data['Model'],
                        y=task_data['Primary_Score'],
                        marker_color=self.color_palette['models'][i % len(self.color_palette['models'])],
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Model Performance Comparison Across All Tasks",
            showlegend=True,
            **self.default_layout
        )
        
        return fig
    
    def generate_clinical_interpretation(self, results: Dict, structured_data: Tuple) -> Dict[str, str]:
        """
        Generate clinical interpretation and insights from model results.
        
        Args:
            results: Dictionary with model results
            structured_data: Tuple of (X, y_dict, feature_names)
            
        Returns:
            Dictionary with clinical interpretations
        """
        X, y_dict, feature_names = structured_data
        interpretation = {}
        
        # Overall model performance summary
        interpretation['Model Performance Summary'] = self._generate_performance_summary(results)
        
        # Feature importance insights
        interpretation['Key Predictive Features'] = self._analyze_feature_importance(results, feature_names)
        
        # Clinical insights by task
        interpretation['Clinical Task Insights'] = self._generate_task_insights(results)
        
        # Recommendations for clinical practice
        interpretation['Clinical Practice Recommendations'] = self._generate_clinical_recommendations(results)
        
        # Model limitations and considerations
        interpretation['Limitations and Considerations'] = self._generate_limitations(results, X.shape)
        
        return interpretation
    
    def _generate_performance_summary(self, results: Dict) -> str:
        """Generate overall performance summary."""
        summary_parts = []
        
        for task_name, task_results in results.items():
            best_score = -np.inf
            best_model = None
            
            for model_name, model_result in task_results.items():
                if 'classification' in model_result:
                    score = model_result['classification'].get('auc_roc', 
                           model_result['classification'].get('f1_score', 0))
                    metric = 'AUC-ROC' if 'auc_roc' in model_result['classification'] else 'F1-Score'
                elif 'regression' in model_result:
                    score = model_result['regression'].get('r2', 0)
                    metric = 'R²'
                else:
                    continue
                
                if score > best_score:
                    best_score = score
                    best_model = model_name
            
            if best_model:
                summary_parts.append(f"**{task_name}**: Best model is {best_model} with {metric} = {best_score:.3f}")
        
        if not summary_parts:
            return "No valid model results found for performance summary."
        
        return "\n\n".join(summary_parts)
    
    def _analyze_feature_importance(self, results: Dict, feature_names: List[str]) -> str:
        """Analyze and summarize feature importance across models."""
        # Aggregate feature importance across all models
        feature_importance_agg = {}
        
        for task_name, task_results in results.items():
            for model_name, model_result in task_results.items():
                importance = None
                
                if 'classification' in model_result:
                    importance = model_result['classification'].get('feature_importance')
                elif 'regression' in model_result:
                    importance = model_result['regression'].get('feature_importance')
                
                if importance:
                    for feature, score in importance.items():
                        if feature not in feature_importance_agg:
                            feature_importance_agg[feature] = []
                        feature_importance_agg[feature].append(score)
        
        # Calculate average importance
        avg_importance = {}
        for feature, scores in feature_importance_agg.items():
            avg_importance[feature] = np.mean(scores)
        
        # Get top features
        top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if not top_features:
            return "No feature importance data available."
        
        # Generate clinical interpretation
        insights = ["**Most Important Predictive Features:**\n"]
        
        for i, (feature, importance) in enumerate(top_features, 1):
            clinical_meaning = self._interpret_feature_clinically(feature)
            insights.append(f"{i}. **{feature}** (Importance: {importance:.3f}): {clinical_meaning}")
        
        return "\n".join(insights)
    
    def _interpret_feature_clinically(self, feature: str) -> str:
        """Provide clinical interpretation of a feature."""
        feature_lower = feature.lower()
        
        # Clinical interpretations for common features
        if 'age' in feature_lower:
            return "Age is a critical factor in hand surgery outcomes, affecting healing capacity and complication risk."
        elif 'zone' in feature_lower:
            return "Tendon injury zone classification directly impacts surgical complexity and functional outcomes."
        elif 'grip' in feature_lower:
            return "Grip strength is a primary functional outcome measure in hand surgery."
        elif 'complication' in feature_lower:
            return "Previous complications are strong predictors of future surgical risks."
        elif 'nerve' in feature_lower:
            return "Nerve involvement significantly affects recovery time and functional outcomes."
        elif 'fracture' in feature_lower:
            return "Fracture type and location influence surgical approach and healing time."
        elif 'intervention' in feature_lower:
            return "Specific surgical interventions impact complexity and success rates."
        elif 'motion' in feature_lower or 'rom' in feature_lower:
            return "Range of motion is essential for functional hand use and quality of life."
        else:
            return "This factor contributes to predicting clinical outcomes in hand surgery."
    
    def _generate_task_insights(self, results: Dict) -> str:
        """Generate insights specific to each clinical task."""
        insights = []
        
        for task_name, task_results in results.items():
            if task_name == 'Complication Risk':
                insights.append(self._complication_insights(task_results))
            elif task_name == 'Return to Function':
                insights.append(self._return_function_insights(task_results))
            elif task_name == 'Surgical Success':
                insights.append(self._surgical_success_insights(task_results))
        
        return "\n\n".join(insights)
    
    def _complication_insights(self, task_results: Dict) -> str:
        """Generate insights for complication risk prediction."""
        best_auc = 0
        best_model = None
        
        for model_name, model_result in task_results.items():
            if 'classification' in model_result:
                auc = model_result['classification'].get('auc_roc', 0)
                if auc > best_auc:
                    best_auc = auc
                    best_model = model_name
        
        insight = f"**Complication Risk Prediction**: "
        
        if best_auc > 0.8:
            insight += f"Excellent predictive performance (AUC = {best_auc:.3f}) suggests reliable complication risk assessment. "
        elif best_auc > 0.7:
            insight += f"Good predictive performance (AUC = {best_auc:.3f}) indicates useful clinical decision support. "
        else:
            insight += f"Moderate predictive performance (AUC = {best_auc:.3f}) suggests additional factors may be needed. "
        
        insight += "Early identification of high-risk patients can guide preventive interventions and informed consent."
        
        return insight
    
    def _return_function_insights(self, task_results: Dict) -> str:
        """Generate insights for return to function prediction."""
        best_r2 = -np.inf
        best_model = None
        
        for model_name, model_result in task_results.items():
            if 'regression' in model_result:
                r2 = model_result['regression'].get('r2', -np.inf)
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model_name
        
        insight = f"**Return to Function Prediction**: "
        
        if best_r2 > 0.7:
            insight += f"Strong predictive capability (R² = {best_r2:.3f}) for estimating recovery time. "
        elif best_r2 > 0.5:
            insight += f"Moderate predictive capability (R² = {best_r2:.3f}) provides useful estimates. "
        else:
            insight += f"Limited predictive capability (R² = {best_r2:.3f}) suggests high variability in recovery times. "
        
        insight += "Accurate recovery time prediction helps with patient expectations and return-to-work planning."
        
        return insight
    
    def _surgical_success_insights(self, task_results: Dict) -> str:
        """Generate insights for surgical success prediction."""
        best_f1 = 0
        best_model = None
        
        for model_name, model_result in task_results.items():
            if 'classification' in model_result:
                f1 = model_result['classification'].get('f1_score', 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model_name
        
        insight = f"**Surgical Success Prediction**: "
        
        if best_f1 > 0.8:
            insight += f"High accuracy (F1 = {best_f1:.3f}) in predicting surgical outcomes. "
        elif best_f1 > 0.7:
            insight += f"Good accuracy (F1 = {best_f1:.3f}) provides valuable outcome predictions. "
        else:
            insight += f"Moderate accuracy (F1 = {best_f1:.3f}) indicates outcome complexity. "
        
        insight += "Preoperative success prediction aids in surgical planning and patient counseling."
        
        return insight
    
    def _generate_clinical_recommendations(self, results: Dict) -> str:
        """Generate recommendations for clinical practice."""
        recommendations = [
            "**Clinical Practice Recommendations:**\n",
            "1. **Risk Stratification**: Use complication risk models to identify high-risk patients requiring enhanced monitoring and preventive measures.",
            "",
            "2. **Patient Counseling**: Leverage outcome predictions to provide patients with evidence-based expectations about recovery times and success rates.",
            "",
            "3. **Surgical Planning**: Consider predictive factors when planning surgical approach and postoperative care protocols.",
            "",
            "4. **Quality Improvement**: Monitor actual outcomes against predictions to identify areas for clinical process improvement.",
            "",
            "5. **Resource Allocation**: Use recovery time predictions for better scheduling and resource planning in rehabilitation services."
        ]
        
        return "\n".join(recommendations)
    
    def _generate_limitations(self, results: Dict, data_shape: Tuple[int, int]) -> str:
        """Generate limitations and considerations."""
        n_samples, n_features = data_shape
        
        limitations = [
            "**Limitations and Considerations:**\n",
            f"1. **Sample Size**: Analysis based on {n_samples} abstracts. Larger datasets may improve model robustness.",
            "",
            "2. **Literature Bias**: Results reflect published literature which may have publication bias toward positive outcomes.",
            "",
            "3. **Feature Extraction**: NLP-based feature extraction may miss nuanced clinical details present in full text articles.",
            "",
            "4. **Temporal Factors**: Models may not account for evolving surgical techniques and technologies over time.",
            "",
            "5. **External Validation**: Models should be validated on independent datasets before clinical implementation.",
            "",
            "6. **Clinical Context**: Predictions should complement, not replace, clinical judgment and patient-specific factors."
        ]
        
        return "\n".join(limitations)
    
    def create_confidence_distribution_plot(self, processed_data: List[Dict]) -> go.Figure:
        """
        Create a distribution plot of confidence scores from NLP processing.
        
        Args:
            processed_data: List of processed abstracts with confidence scores
            
        Returns:
            Plotly figure showing confidence score distribution
        """
        if not processed_data:
            return None
        
        confidence_scores = [item['confidence_score'] for item in processed_data]
        
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=confidence_scores,
            nbinsx=20,
            name='Confidence Distribution',
            marker_color=self.color_palette['primary'],
            opacity=0.7
        ))
        
        # Add mean line
        mean_confidence = np.mean(confidence_scores)
        fig.add_vline(
            x=mean_confidence,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_confidence:.3f}"
        )
        
        # Update layout
        fig.update_layout(
            title='Distribution of NLP Confidence Scores',
            xaxis_title='Confidence Score',
            yaxis_title='Frequency',
            **self.default_layout
        )
        
        return fig
    
    def create_entity_extraction_summary(self, processed_data: List[Dict]) -> go.Figure:
        """
        Create a summary visualization of entity extraction results.
        
        Args:
            processed_data: List of processed abstracts
            
        Returns:
            Plotly figure showing entity extraction statistics
        """
        if not processed_data:
            return None
        
        # Count entities by category
        entity_counts = {}
        for item in processed_data:
            entities = item.get('entities', {})
            for category, entity_list in entities.items():
                if category not in entity_counts:
                    entity_counts[category] = 0
                entity_counts[category] += len(entity_list)
        
        if not entity_counts:
            return None
        
        # Create bar chart
        categories = list(entity_counts.keys())
        counts = list(entity_counts.values())
        
        fig = go.Figure(go.Bar(
            x=categories,
            y=counts,
            marker_color=self.color_palette['primary']
        ))
        
        # Update layout
        fig.update_layout(
            title='Entity Extraction Summary by Category',
            xaxis_title='Entity Category',
            yaxis_title='Total Entities Extracted',
            **self.default_layout
        )
        
        return fig
