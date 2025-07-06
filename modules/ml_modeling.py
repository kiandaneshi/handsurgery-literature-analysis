import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
from typing import Dict, List, Tuple, Optional, Any
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

class MLModeler:
    """
    Handles machine learning model training and evaluation for clinical outcome prediction.
    Implements multiple algorithms for classification and regression tasks.
    """
    
    def __init__(self, test_size: float = 0.2, cv_folds: int = 5, random_state: int = 42):
        """
        Initialize ML modeler.
        
        Args:
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Data preprocessing
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
        # Model configurations
        self.classification_models = {
            'Logistic Regression': LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                class_weight='balanced',
                max_depth=10
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                eval_metric='logloss'
            )
        }
        
        self.regression_models = {
            'XGBoost': xgb.XGBRegressor(
                random_state=random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1
            ),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
                max_depth=10
            )
        }
        
        # Task configurations
        self.task_configs = {
            'Complication Risk': {'type': 'classification', 'target': 'complication_risk'},
            'Return to Function': {'type': 'regression', 'target': 'return_to_function'},
            'Surgical Success': {'type': 'classification', 'target': 'surgical_success'}
        }
    
    def train_models(self, structured_data: Tuple, models_to_train: List[str],
                    tasks_to_train: List[str]) -> Optional[Dict]:
        """
        Train machine learning models for specified tasks.
        
        Args:
            structured_data: Tuple of (X, y_dict, feature_names)
            models_to_train: List of model names to train
            tasks_to_train: List of task names to train
            
        Returns:
            Dictionary with training results or None if failed
        """
        try:
            X, y_dict, feature_names = structured_data
            
            # Preprocess features
            X_processed = self._preprocess_features(X)
            
            results = {}
            
            for task_name in tasks_to_train:
                st.info(f"Training models for: {task_name}")
                
                if task_name not in self.task_configs:
                    st.warning(f"Unknown task: {task_name}")
                    continue
                
                task_config = self.task_configs[task_name]
                target_key = task_config['target']
                task_type = task_config['type']
                
                # Get target variable
                y = y_dict.get(target_key)
                if y is None:
                    st.warning(f"No target variable found for {task_name}")
                    continue
                
                # Remove samples with missing targets
                valid_indices = ~np.isnan(y)
                if np.sum(valid_indices) < 10:
                    st.warning(f"Insufficient valid samples for {task_name} ({np.sum(valid_indices)})")
                    continue
                
                X_task = X_processed[valid_indices]
                y_task = y[valid_indices]
                
                # Train models for this task
                task_results = self._train_task_models(
                    X_task, y_task, task_type, models_to_train, feature_names, task_name
                )
                
                if task_results:
                    results[task_name] = task_results
            
            return results if results else None
            
        except Exception as e:
            st.error(f"Error in model training: {str(e)}")
            return None
    
    def _preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess features for machine learning.
        
        Args:
            X: Raw feature matrix
            
        Returns:
            Preprocessed feature matrix
        """
        # Handle missing values
        X_imputed = self.imputer.fit_transform(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        return X_scaled
    
    def _train_task_models(self, X: np.ndarray, y: np.ndarray, task_type: str,
                          models_to_train: List[str], feature_names: List[str],
                          task_name: str) -> Dict:
        """
        Train models for a specific task.
        
        Args:
            X: Feature matrix
            y: Target variable
            task_type: 'classification' or 'regression'
            models_to_train: List of model names
            feature_names: List of feature names
            task_name: Name of the task
            
        Returns:
            Dictionary with model results
        """
        results = {}
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y if task_type == 'classification' and len(np.unique(y)) > 1 else None
        )
        
        # Select appropriate models
        if task_type == 'classification':
            model_dict = self.classification_models
        else:
            model_dict = self.regression_models
        
        for model_name in models_to_train:
            if model_name not in model_dict:
                st.warning(f"Model {model_name} not available for {task_type}")
                continue
            
            try:
                st.info(f"Training {model_name} for {task_name}")
                
                # Train model
                model = model_dict[model_name]
                model.fit(X_train, y_train)
                
                # Evaluate model
                model_results = self._evaluate_model(
                    model, X_train, X_test, y_train, y_test, task_type, feature_names
                )
                
                # Cross-validation
                cv_scores = self._cross_validate_model(model, X, y, task_type)
                model_results['cv_score_mean'] = np.mean(cv_scores)
                model_results['cv_score_std'] = np.std(cv_scores)
                model_results['cv_scores'] = cv_scores
                
                # Store trained model
                model_results['trained_model'] = model
                
                results[model_name] = {task_type: model_results}
                
                st.success(f"✅ {model_name} trained successfully")
                
            except Exception as e:
                st.error(f"Error training {model_name}: {str(e)}")
                continue
        
        return results
    
    def _evaluate_model(self, model: Any, X_train: np.ndarray, X_test: np.ndarray,
                       y_train: np.ndarray, y_test: np.ndarray, task_type: str,
                       feature_names: List[str]) -> Dict:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets
            task_type: 'classification' or 'regression'
            feature_names: List of feature names
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {}
        
        # Predictions
        y_pred = model.predict(X_test)
        
        if task_type == 'classification':
            # Classification metrics
            results['accuracy'] = accuracy_score(y_test, y_pred)
            results['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            results['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            results['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            results['sensitivity'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
            results['specificity'] = self._calculate_specificity(y_test, y_pred)
            
            # ROC AUC (if binary classification)
            if len(np.unique(y_test)) == 2:
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    results['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
                    results['y_test'] = y_test
                    results['y_pred_proba'] = y_pred_proba
                    
                    # ROC curve data
                    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                    results['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
                except:
                    results['auc_roc'] = 0.5
            else:
                results['auc_roc'] = np.nan
            
            # Confusion matrix
            results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
            
        else:
            # Regression metrics
            results['r2'] = r2_score(y_test, y_pred)
            results['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            results['mae'] = mean_absolute_error(y_test, y_pred)
            results['y_test'] = y_test
            results['y_pred'] = y_pred
        
        # Feature importance
        results['feature_importance'] = self._get_feature_importance(model, feature_names)
        
        return results
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity for binary classification."""
        try:
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                return specificity
            else:
                return np.nan
        except:
            return np.nan
    
    def _cross_validate_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                             task_type: str) -> np.ndarray:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to cross-validate
            X: Feature matrix
            y: Target variable
            task_type: 'classification' or 'regression'
            
        Returns:
            Array of cross-validation scores
        """
        try:
            if task_type == 'classification':
                # Use stratified k-fold for classification
                if len(np.unique(y)) > 1:
                    cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                    scoring = 'roc_auc' if len(np.unique(y)) == 2 else 'f1_weighted'
                else:
                    return np.array([0.0] * self.cv_folds)
            else:
                # Use regular k-fold for regression
                cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                scoring = 'r2'
            
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            return scores
            
        except Exception as e:
            st.warning(f"Cross-validation failed: {str(e)}")
            return np.array([0.0] * self.cv_folds)
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """
        Extract feature importance from trained model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        importance = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                if len(model.coef_.shape) > 1:
                    importances = np.abs(model.coef_[0])
                else:
                    importances = np.abs(model.coef_)
            else:
                # Default: equal importance
                importances = np.ones(len(feature_names)) / len(feature_names)
            
            # Create importance dictionary
            for i, name in enumerate(feature_names):
                if i < len(importances):
                    importance[name] = float(importances[i])
                else:
                    importance[name] = 0.0
                    
        except Exception as e:
            st.warning(f"Could not extract feature importance: {str(e)}")
            # Default equal importance
            for name in feature_names:
                importance[name] = 1.0 / len(feature_names)
        
        return importance
    
    def get_model_summary(self, results: Dict) -> pd.DataFrame:
        """
        Create a summary DataFrame of all model results.
        
        Args:
            results: Dictionary with model results
            
        Returns:
            DataFrame with model performance summary
        """
        summary_data = []
        
        for task_name, task_results in results.items():
            for model_name, model_result in task_results.items():
                
                if 'classification' in model_result:
                    metrics = model_result['classification']
                    summary_data.append({
                        'Task': task_name,
                        'Model': model_name,
                        'Type': 'Classification',
                        'Primary_Metric': metrics.get('auc_roc', metrics.get('f1_score', 0)),
                        'Accuracy': metrics.get('accuracy', 0),
                        'F1_Score': metrics.get('f1_score', 0),
                        'AUC_ROC': metrics.get('auc_roc', 0),
                        'CV_Mean': metrics.get('cv_score_mean', 0),
                        'CV_Std': metrics.get('cv_score_std', 0)
                    })
                
                elif 'regression' in model_result:
                    metrics = model_result['regression']
                    summary_data.append({
                        'Task': task_name,
                        'Model': model_name,
                        'Type': 'Regression',
                        'Primary_Metric': metrics.get('r2', 0),
                        'R2': metrics.get('r2', 0),
                        'RMSE': metrics.get('rmse', 0),
                        'MAE': metrics.get('mae', 0),
                        'CV_Mean': metrics.get('cv_score_mean', 0),
                        'CV_Std': metrics.get('cv_score_std', 0)
                    })
        
        return pd.DataFrame(summary_data)
    
    def get_best_models(self, results: Dict) -> Dict[str, Tuple[str, float]]:
        """
        Identify best performing model for each task.
        
        Args:
            results: Dictionary with model results
            
        Returns:
            Dictionary mapping task names to (best_model_name, best_score)
        """
        best_models = {}
        
        for task_name, task_results in results.items():
            best_score = -np.inf
            best_model = None
            
            for model_name, model_result in task_results.items():
                
                if 'classification' in model_result:
                    # Use AUC-ROC if available, else F1-score
                    score = model_result['classification'].get('auc_roc')
                    if score is None or np.isnan(score):
                        score = model_result['classification'].get('f1_score', 0)
                
                elif 'regression' in model_result:
                    # Use R² for regression
                    score = model_result['regression'].get('r2', -np.inf)
                
                else:
                    continue
                
                if score > best_score:
                    best_score = score
                    best_model = model_name
            
            if best_model:
                best_models[task_name] = (best_model, best_score)
        
        return best_models
    
    def predict_with_best_models(self, results: Dict, X_new: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make predictions using the best model for each task.
        
        Args:
            results: Dictionary with model results
            X_new: New feature matrix for prediction
            
        Returns:
            Dictionary with predictions for each task
        """
        predictions = {}
        best_models = self.get_best_models(results)
        
        # Preprocess new data
        X_processed = self.scaler.transform(self.imputer.transform(X_new))
        
        for task_name, (best_model_name, _) in best_models.items():
            task_results = results[task_name]
            model_result = task_results[best_model_name]
            
            # Get the appropriate model type result
            if 'classification' in model_result:
                trained_model = model_result['classification']['trained_model']
                predictions[task_name] = trained_model.predict(X_processed)
            elif 'regression' in model_result:
                trained_model = model_result['regression']['trained_model']
                predictions[task_name] = trained_model.predict(X_processed)
        
        return predictions
