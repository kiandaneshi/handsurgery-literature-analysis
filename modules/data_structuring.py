import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple, Optional
import streamlit as st
import re
from datetime import datetime

class DataStructurer:
    """
    Converts NLP-extracted entities into structured features for machine learning.
    Handles feature engineering, encoding, and target variable creation.
    """
    
    def __init__(self, encoding_method: str = "One-Hot Encoding", 
                 include_text_features: bool = True):
        """
        Initialize data structurer.
        
        Args:
            encoding_method: Method for categorical encoding
            include_text_features: Whether to include TF-IDF text features
        """
        self.encoding_method = encoding_method
        self.include_text_features = include_text_features
        
        # Encoders
        self.label_encoders = {}
        self.onehot_encoders = {}
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        # Feature mapping dictionaries
        self.age_group_mapping = {
            'pediatric': 0, 'child': 0, 'children': 0, 'infant': 0, 'baby': 0,
            'adult': 1, 'grown': 1,
            'elderly': 2, 'geriatric': 2, 'senior': 2, 'aged': 2
        }
        
        # Surgical intervention mappings
        self.intervention_types = [
            'grafting', 'repair', 'fixation', 'pinning', 'plating',
            'decompression', 'release', 'splinting', 'therapy'
        ]
        
        # Outcome indicators
        self.outcome_indicators = {
            'complications': ['complication', 'adverse', 'infection', 'rupture', 'failure'],
            'success': ['successful', 'good', 'excellent', 'improved', 'recovered'],
            'return_to_work': ['return to work', 'rtw', 'occupational'],
            'functional_recovery': ['function', 'grip', 'strength', 'motion', 'rom']
        }
    
    def structure_data(self, processed_data: List[Dict], 
                      abstracts_df: pd.DataFrame) -> Optional[Tuple]:
        """
        Convert processed NLP data into ML-ready structured features.
        
        Args:
            processed_data: List of processed abstracts with entities
            abstracts_df: Original abstracts DataFrame
            
        Returns:
            Tuple of (X, y_dict, feature_names) or None if failed
        """
        try:
            st.info("Converting NLP outputs to structured features...")
            
            # Create feature DataFrame
            features_df = self._create_features_dataframe(processed_data, abstracts_df)
            
            if features_df.empty:
                st.error("No valid features could be extracted from the data.")
                return None
            
            # Extract target variables
            target_variables = self._extract_target_variables(features_df, processed_data)
            
            # Prepare feature matrix
            X, feature_names = self._prepare_feature_matrix(features_df)
            
            st.success(f"Structured data created: {X.shape[0]} samples × {X.shape[1]} features")
            
            return X, target_variables, feature_names
            
        except Exception as e:
            st.error(f"Error in data structuring: {str(e)}")
            return None
    
    def _create_features_dataframe(self, processed_data: List[Dict], 
                                 abstracts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create initial features DataFrame from processed entities.
        
        Args:
            processed_data: Processed NLP data
            abstracts_df: Original abstracts
            
        Returns:
            DataFrame with extracted features
        """
        features_list = []
        
        # Create PMID to abstract mapping
        abstract_lookup = abstracts_df.set_index('pmid').to_dict('index')
        
        for item in processed_data:
            pmid = item['pmid']
            entities = item['entities']
            
            # Get original abstract data
            abstract_info = abstract_lookup.get(pmid, {})
            
            # Initialize feature dict
            features = {
                'pmid': pmid,
                'confidence_score': item['confidence_score'],
                'year': abstract_info.get('year'),
                'journal': abstract_info.get('journal', ''),
                'abstract_length': len(item.get('abstract', '')),
                'title_length': len(item.get('title', ''))
            }
            
            # Extract demographic features
            features.update(self._extract_demographic_features(entities))
            
            # Extract injury/procedure features
            features.update(self._extract_injury_procedure_features(entities))
            
            # Extract surgical intervention features
            features.update(self._extract_surgical_features(entities))
            
            # Extract outcome features
            features.update(self._extract_outcome_features(entities, item))
            
            # Extract text-based features
            if self.include_text_features:
                features.update(self._extract_text_features(item))
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _extract_demographic_features(self, entities: Dict[str, List[str]]) -> Dict:
        """Extract demographic features from entities."""
        features = {}
        
        demographics = entities.get('demographics', [])
        demo_text = ' '.join(demographics).lower()
        
        # Age group encoding
        age_group = self._determine_age_group(demo_text)
        features['age_group'] = age_group
        
        # Gender detection
        gender = 'unknown'
        if any(term in demo_text for term in ['male', 'men', 'man']):
            if any(term in demo_text for term in ['female', 'women', 'woman']):
                gender = 'mixed'
            else:
                gender = 'male'
        elif any(term in demo_text for term in ['female', 'women', 'woman']):
            gender = 'female'
        
        features['gender'] = gender
        
        # Population type
        pop_type = 'general'
        if any(term in demo_text for term in ['pediatric', 'child', 'infant']):
            pop_type = 'pediatric'
        elif any(term in demo_text for term in ['elderly', 'geriatric', 'senior']):
            pop_type = 'elderly'
        
        features['population_type'] = pop_type
        
        return features
    
    def _determine_age_group(self, text: str) -> int:
        """Determine age group from text."""
        # Look for explicit age group terms
        for term, code in self.age_group_mapping.items():
            if term in text:
                return code
        
        # Look for age ranges
        age_pattern = r'(\d+)\s*[-–]\s*(\d+)\s*(year|yr|y)s?\s*(old|age)'
        match = re.search(age_pattern, text)
        if match:
            start_age = int(match.group(1))
            if start_age <= 18:
                return 0  # pediatric
            elif start_age <= 65:
                return 1  # adult
            else:
                return 2  # elderly
        
        # Look for mean age
        mean_age_pattern = r'(mean|average)\s*age\s*(\d+)'
        match = re.search(mean_age_pattern, text)
        if match:
            age = int(match.group(2))
            if age <= 18:
                return 0
            elif age <= 65:
                return 1
            else:
                return 2
        
        return 1  # default to adult
    
    def _extract_injury_procedure_features(self, entities: Dict[str, List[str]]) -> Dict:
        """Extract injury and procedure-related features."""
        features = {}
        
        injury_procedure = entities.get('injury_procedure', [])
        injury_text = ' '.join(injury_procedure).lower()
        
        # Tendon zone
        zone_features = {}
        for zone in ['i', 'ii', 'iii', 'iv', 'v', '1', '2', '3', '4', '5']:
            zone_features[f'zone_{zone}'] = 1 if f'zone {zone}' in injury_text else 0
        features.update(zone_features)
        
        # Fracture type
        fracture_types = ['metacarpal', 'phalangeal', 'scaphoid', 'radius', 'ulna']
        for ftype in fracture_types:
            features[f'fracture_{ftype}'] = 1 if ftype in injury_text else 0
        
        # Nerve type
        nerve_types = ['median', 'ulnar', 'radial', 'digital']
        for ntype in nerve_types:
            features[f'nerve_{ntype}'] = 1 if ntype in injury_text else 0
        
        # Anatomical location
        locations = ['thumb', 'index', 'middle', 'ring', 'little', 'wrist', 'forearm']
        for location in locations:
            features[f'location_{location}'] = 1 if location in injury_text else 0
        
        return features
    
    def _extract_surgical_features(self, entities: Dict[str, List[str]]) -> Dict:
        """Extract surgical intervention features."""
        features = {}
        
        surgical = entities.get('surgical_interventions', [])
        surgical_text = ' '.join(surgical).lower()
        
        # Binary features for each intervention type
        for intervention in self.intervention_types:
            features[f'intervention_{intervention}'] = 1 if intervention in surgical_text else 0
        
        # Count total interventions
        features['total_interventions'] = sum(features[f'intervention_{i}'] for i in self.intervention_types)
        
        return features
    
    def _extract_outcome_features(self, entities: Dict[str, List[str]], 
                                item: Dict) -> Dict:
        """Extract clinical outcome features."""
        features = {}
        
        outcomes = entities.get('clinical_outcomes', [])
        outcome_text = ' '.join(outcomes).lower()
        
        # Add abstract text for outcome detection
        full_text = f"{item.get('title', '')} {item.get('abstract', '')}".lower()
        
        # Outcome indicators
        for outcome_type, indicators in self.outcome_indicators.items():
            has_outcome = any(indicator in full_text for indicator in indicators)
            features[f'has_{outcome_type}'] = 1 if has_outcome else 0
        
        # Specific outcome measures
        outcome_measures = ['grip_strength', 'range_of_motion', 'pain', 'dash', 'quickdash']
        for measure in outcome_measures:
            features[f'measured_{measure}'] = 1 if measure.replace('_', ' ') in outcome_text else 0
        
        # Extract numerical values if possible
        features.update(self._extract_numerical_outcomes(full_text))
        
        return features
    
    def _extract_numerical_outcomes(self, text: str) -> Dict:
        """Extract numerical outcome values from text."""
        features = {}
        
        # Grip strength (kg or lbs)
        grip_pattern = r'grip\s*strength\s*(\d+(?:\.\d+)?)\s*(kg|lb|pound)'
        match = re.search(grip_pattern, text)
        if match:
            value = float(match.group(1))
            if 'lb' in match.group(2) or 'pound' in match.group(2):
                value = value * 0.453592  # Convert to kg
            features['grip_strength_value'] = value
        else:
            features['grip_strength_value'] = np.nan
        
        # Range of motion (degrees)
        rom_pattern = r'range\s*of\s*motion\s*(\d+(?:\.\d+)?)\s*(?:degree|°)'
        match = re.search(rom_pattern, text)
        features['rom_value'] = float(match.group(1)) if match else np.nan
        
        # Pain scores (0-10 scale typically)
        pain_pattern = r'pain\s*(?:score|level)\s*(\d+(?:\.\d+)?)'
        match = re.search(pain_pattern, text)
        features['pain_score'] = float(match.group(1)) if match else np.nan
        
        # Return to work time (days/weeks/months)
        rtw_pattern = r'return\s*to\s*work\s*(\d+(?:\.\d+)?)\s*(day|week|month)'
        match = re.search(rtw_pattern, text)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            if 'week' in unit:
                value *= 7
            elif 'month' in unit:
                value *= 30
            features['rtw_days'] = value
        else:
            features['rtw_days'] = np.nan
        
        return features
    
    def _extract_text_features(self, item: Dict) -> Dict:
        """Extract TF-IDF features from text."""
        features = {}
        
        # Combine title and abstract
        text = f"{item.get('title', '')} {item.get('abstract', '')}"
        
        # Extract key clinical terms frequency
        clinical_terms = [
            'surgery', 'treatment', 'outcome', 'patient', 'result',
            'hand', 'finger', 'tendon', 'nerve', 'fracture'
        ]
        
        for term in clinical_terms:
            count = text.lower().count(term)
            features[f'term_freq_{term}'] = count
        
        # Text complexity features
        features['avg_sentence_length'] = len(text.split()) / max(1, text.count('.'))
        features['unique_word_ratio'] = len(set(text.lower().split())) / max(1, len(text.split()))
        
        return features
    
    def _extract_target_variables(self, features_df: pd.DataFrame, 
                                processed_data: List[Dict]) -> Dict:
        """Extract target variables for ML models."""
        targets = {}
        
        # Complication Risk (Binary Classification)
        targets['complication_risk'] = self._create_complication_target(features_df, processed_data)
        
        # Return to Function (Regression)
        targets['return_to_function'] = self._create_return_function_target(features_df)
        
        # Surgical Success (Binary Classification)
        targets['surgical_success'] = self._create_success_target(features_df, processed_data)
        
        return targets
    
    def _create_complication_target(self, features_df: pd.DataFrame, 
                                  processed_data: List[Dict]) -> np.ndarray:
        """Create binary complication risk target."""
        complications = []
        
        for i, item in enumerate(processed_data):
            full_text = f"{item.get('title', '')} {item.get('abstract', '')}".lower()
            
            # Look for complication indicators
            complication_terms = [
                'complication', 'adverse', 'infection', 'rupture', 'failure',
                'reoperation', 'revision', 'dehiscence', 'necrosis'
            ]
            
            has_complication = any(term in full_text for term in complication_terms)
            complications.append(1 if has_complication else 0)
        
        return np.array(complications)
    
    def _create_return_function_target(self, features_df: pd.DataFrame) -> np.ndarray:
        """Create return to function regression target (days)."""
        return features_df['rtw_days'].values
    
    def _create_success_target(self, features_df: pd.DataFrame, 
                             processed_data: List[Dict]) -> np.ndarray:
        """Create binary surgical success target."""
        success = []
        
        for i, item in enumerate(processed_data):
            full_text = f"{item.get('title', '')} {item.get('abstract', '')}".lower()
            
            # Success indicators
            success_terms = [
                'successful', 'good', 'excellent', 'improved', 'recovered',
                'satisfactory', 'effective', 'beneficial'
            ]
            
            # Failure indicators
            failure_terms = [
                'failed', 'poor', 'unsatisfactory', 'ineffective', 'unsuccessful'
            ]
            
            success_score = sum(1 for term in success_terms if term in full_text)
            failure_score = sum(1 for term in failure_terms if term in full_text)
            
            if success_score > failure_score:
                success.append(1)
            elif failure_score > success_score:
                success.append(0)
            else:
                success.append(np.nan)  # Ambiguous cases
        
        return np.array(success, dtype=float)
    
    def _prepare_feature_matrix(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare final feature matrix for ML."""
        # Select features for ML (exclude metadata)
        exclude_cols = ['pmid', 'journal', 'year']  # Keep these as separate metadata
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        X_df = features_df[feature_cols].copy()
        
        # Handle categorical encoding
        categorical_cols = ['gender', 'population_type']
        
        if self.encoding_method == "One-Hot Encoding":
            X_df = pd.get_dummies(X_df, columns=categorical_cols, prefix=categorical_cols)
        else:  # Ordinal encoding
            for col in categorical_cols:
                if col in X_df.columns:
                    le = LabelEncoder()
                    X_df[col] = le.fit_transform(X_df[col].astype(str))
                    self.label_encoders[col] = le
        
        # Convert to numpy array
        X = X_df.values.astype(float)
        feature_names = list(X_df.columns)
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        return X, feature_names
    
    def get_feature_summary(self, X: np.ndarray, feature_names: List[str], 
                          y_dict: Dict) -> Dict:
        """Generate summary statistics about the structured data."""
        summary = {
            'feature_matrix_shape': X.shape,
            'feature_names': feature_names,
            'missing_values_per_feature': {},
            'target_variable_summary': {}
        }
        
        # Feature statistics
        for i, fname in enumerate(feature_names):
            missing_count = np.sum(np.isnan(X[:, i]))
            summary['missing_values_per_feature'][fname] = missing_count
        
        # Target variable statistics
        for target_name, target_values in y_dict.items():
            if target_values is not None:
                valid_values = target_values[~np.isnan(target_values)]
                
                if len(valid_values) == 0:
                    summary['target_variable_summary'][target_name] = {
                        'type': 'no_valid_data',
                        'total_samples': len(target_values),
                        'missing_samples': len(target_values)
                    }
                elif len(np.unique(valid_values)) == 2:  # Binary classification
                    unique_vals, counts = np.unique(valid_values, return_counts=True)
                    summary['target_variable_summary'][target_name] = {
                        'type': 'binary_classification',
                        'class_distribution': dict(zip(unique_vals, counts)),
                        'valid_samples': len(valid_values),
                        'missing_samples': np.sum(np.isnan(target_values))
                    }
                else:  # Regression
                    summary['target_variable_summary'][target_name] = {
                        'type': 'regression',
                        'mean': np.mean(valid_values) if len(valid_values) > 0 else 0,
                        'std': np.std(valid_values) if len(valid_values) > 0 else 0,
                        'min': np.min(valid_values) if len(valid_values) > 0 else 0,
                        'max': np.max(valid_values) if len(valid_values) > 0 else 0,
                        'valid_samples': len(valid_values),
                        'missing_samples': np.sum(np.isnan(target_values))
                    }
        
        return summary
