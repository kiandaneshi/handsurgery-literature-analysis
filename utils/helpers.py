import pandas as pd
import numpy as np
import re
import string
from typing import List, Dict, Any, Optional, Union, Tuple
import streamlit as st
from datetime import datetime, timedelta
import json
import pickle
import os
from pathlib import Path

class DataValidator:
    """Utility class for data validation and quality checks."""
    
    @staticmethod
    def validate_pubmed_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate PubMed data quality and completeness.
        
        Args:
            df: DataFrame with PubMed data
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        if df is None or df.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("DataFrame is empty or None")
            return validation_results
        
        # Required columns
        required_columns = ['pmid', 'title', 'abstract']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check for duplicates
        if 'pmid' in df.columns:
            duplicates = df['pmid'].duplicated().sum()
            if duplicates > 0:
                validation_results['warnings'].append(f"Found {duplicates} duplicate PMIDs")
        
        # Check abstract quality
        if 'abstract' in df.columns:
            short_abstracts = (df['abstract'].str.len() < 50).sum()
            if short_abstracts > 0:
                validation_results['warnings'].append(f"{short_abstracts} abstracts are shorter than 50 characters")
        
        # Check year validity
        if 'year' in df.columns:
            current_year = datetime.now().year
            invalid_years = ((df['year'] < 1990) | (df['year'] > current_year)).sum()
            if invalid_years > 0:
                validation_results['warnings'].append(f"{invalid_years} records have invalid publication years")
        
        # Statistics
        validation_results['statistics'] = {
            'total_records': len(df),
            'unique_pmids': df['pmid'].nunique() if 'pmid' in df.columns else 0,
            'avg_abstract_length': df['abstract'].str.len().mean() if 'abstract' in df.columns else 0,
            'year_range': f"{df['year'].min()}-{df['year'].max()}" if 'year' in df.columns else 'N/A'
        }
        
        return validation_results
    
    @staticmethod
    def validate_nlp_results(processed_data: List[Dict]) -> Dict[str, Any]:
        """
        Validate NLP processing results.
        
        Args:
            processed_data: List of processed abstracts
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        if not processed_data:
            validation_results['is_valid'] = False
            validation_results['errors'].append("No processed data provided")
            return validation_results
        
        # Check required fields
        required_fields = ['pmid', 'entities', 'confidence_score']
        
        missing_fields_count = 0
        low_confidence_count = 0
        empty_entities_count = 0
        
        for item in processed_data:
            # Check required fields
            missing_fields = [field for field in required_fields if field not in item]
            if missing_fields:
                missing_fields_count += 1
            
            # Check confidence scores
            confidence = item.get('confidence_score', 0)
            if confidence < 0.3:
                low_confidence_count += 1
            
            # Check entity extraction
            entities = item.get('entities', {})
            total_entities = sum(len(entity_list) for entity_list in entities.values())
            if total_entities == 0:
                empty_entities_count += 1
        
        # Add warnings
        if missing_fields_count > 0:
            validation_results['warnings'].append(f"{missing_fields_count} records missing required fields")
        
        if low_confidence_count > 0:
            validation_results['warnings'].append(f"{low_confidence_count} records have low confidence scores (<0.3)")
        
        if empty_entities_count > 0:
            validation_results['warnings'].append(f"{empty_entities_count} records have no extracted entities")
        
        # Statistics
        confidence_scores = [item.get('confidence_score', 0) for item in processed_data]
        validation_results['statistics'] = {
            'total_processed': len(processed_data),
            'avg_confidence': np.mean(confidence_scores),
            'min_confidence': np.min(confidence_scores),
            'max_confidence': np.max(confidence_scores),
            'high_confidence_count': sum(1 for score in confidence_scores if score >= 0.7)
        }
        
        return validation_results

class TextProcessor:
    """Utility class for text processing and cleaning."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def extract_numerical_values(text: str, pattern: str, unit: str = None) -> List[float]:
        """
        Extract numerical values from text using regex patterns.
        
        Args:
            text: Input text
            pattern: Regex pattern to match
            unit: Optional unit to look for
            
        Returns:
            List of extracted numerical values
        """
        values = []
        
        if unit:
            full_pattern = f"{pattern}\\s*{unit}"
        else:
            full_pattern = pattern
        
        matches = re.finditer(full_pattern, text, re.IGNORECASE)
        
        for match in matches:
            try:
                # Extract the first captured group (the number)
                value = float(match.group(1))
                values.append(value)
            except (ValueError, IndexError):
                continue
        
        return values
    
    @staticmethod
    def standardize_medical_terms(text: str) -> str:
        """
        Standardize medical terminology in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with standardized medical terms
        """
        # Dictionary of term standardizations
        standardizations = {
            r'\bPIP\b': 'proximal interphalangeal',
            r'\bDIP\b': 'distal interphalangeal',
            r'\bMCP\b': 'metacarpophalangeal',
            r'\bROM\b': 'range of motion',
            r'\bDASH\b': 'disabilities of arm shoulder hand',
            r'\bFDP\b': 'flexor digitorum profundus',
            r'\bFDS\b': 'flexor digitorum superficialis',
            r'\bEPL\b': 'extensor pollicis longus',
            r'\bEDC\b': 'extensor digitorum communis'
        }
        
        for pattern, replacement in standardizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def extract_age_information(text: str) -> Dict[str, Any]:
        """
        Extract age-related information from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with age information
        """
        age_info = {
            'age_mentioned': False,
            'age_values': [],
            'age_group': None,
            'mean_age': None
        }
        
        # Age range patterns
        age_range_pattern = r'(\d+)\s*[-–]\s*(\d+)\s*(year|yr|y)s?\s*(old|age)'
        age_ranges = re.finditer(age_range_pattern, text, re.IGNORECASE)
        
        for match in age_ranges:
            age_info['age_mentioned'] = True
            start_age = int(match.group(1))
            end_age = int(match.group(2))
            age_info['age_values'].extend([start_age, end_age])
        
        # Mean age pattern
        mean_age_pattern = r'(mean|average)\s*age\s*(\d+(?:\.\d+)?)'
        mean_match = re.search(mean_age_pattern, text, re.IGNORECASE)
        
        if mean_match:
            age_info['age_mentioned'] = True
            age_info['mean_age'] = float(mean_match.group(2))
            age_info['age_values'].append(age_info['mean_age'])
        
        # Determine age group
        if age_info['age_values']:
            avg_age = np.mean(age_info['age_values'])
            if avg_age <= 18:
                age_info['age_group'] = 'pediatric'
            elif avg_age <= 65:
                age_info['age_group'] = 'adult'
            else:
                age_info['age_group'] = 'elderly'
        
        return age_info

class FileManager:
    """Utility class for file operations and data persistence."""
    
    @staticmethod
    def save_data_to_file(data: Any, filepath: str, format: str = 'pickle') -> bool:
        """
        Save data to file in specified format.
        
        Args:
            data: Data to save
            filepath: Path to save file
            format: Format ('pickle', 'json', 'csv')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            
            elif format == 'json':
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            
            elif format == 'csv':
                if isinstance(data, pd.DataFrame):
                    data.to_csv(filepath, index=False)
                else:
                    return False
            
            else:
                return False
            
            return True
            
        except Exception as e:
            st.error(f"Error saving file {filepath}: {str(e)}")
            return False
    
    @staticmethod
    def load_data_from_file(filepath: str, format: str = 'pickle') -> Any:
        """
        Load data from file.
        
        Args:
            filepath: Path to load file
            format: Format ('pickle', 'json', 'csv')
            
        Returns:
            Loaded data or None if failed
        """
        try:
            if not os.path.exists(filepath):
                return None
            
            if format == 'pickle':
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            
            elif format == 'json':
                with open(filepath, 'r') as f:
                    return json.load(f)
            
            elif format == 'csv':
                return pd.read_csv(filepath)
            
            else:
                return None
                
        except Exception as e:
            st.error(f"Error loading file {filepath}: {str(e)}")
            return None
    
    @staticmethod
    def create_backup(data: Any, base_filename: str) -> str:
        """
        Create a timestamped backup of data.
        
        Args:
            data: Data to backup
            base_filename: Base filename for backup
            
        Returns:
            Path to backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{base_filename}_{timestamp}.pkl"
        backup_path = f"backups/{backup_filename}"
        
        if FileManager.save_data_to_file(data, backup_path, 'pickle'):
            return backup_path
        else:
            return None

class ProgressTracker:
    """Utility class for tracking and displaying progress."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps
            description: Description of the process
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
    
    def update(self, step: int = None, message: str = None):
        """
        Update progress.
        
        Args:
            step: Current step number (optional)
            message: Status message (optional)
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        # Update progress bar
        progress = min(self.current_step / self.total_steps, 1.0)
        self.progress_bar.progress(progress)
        
        # Update status message
        elapsed_time = datetime.now() - self.start_time
        if progress > 0:
            estimated_total = elapsed_time / progress
            remaining_time = estimated_total - elapsed_time
            time_str = f"ETA: {str(remaining_time).split('.')[0]}"
        else:
            time_str = "Calculating..."
        
        if message:
            status_message = f"{self.description}: {message} ({self.current_step}/{self.total_steps}) - {time_str}"
        else:
            status_message = f"{self.description}: {self.current_step}/{self.total_steps} - {time_str}"
        
        self.status_text.text(status_message)
    
    def complete(self, message: str = "Complete"):
        """Mark progress as complete."""
        self.progress_bar.progress(1.0)
        elapsed_time = datetime.now() - self.start_time
        self.status_text.text(f"{self.description}: {message} (Total time: {str(elapsed_time).split('.')[0]})")

class ConfigManager:
    """Utility class for managing application configuration."""
    
    @staticmethod
    def load_config(config_path: str = "config.json") -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'api_settings': {
                'max_retries': 3,
                'timeout': 30,
                'rate_limit_delay': 0.1
            },
            'nlp_settings': {
                'confidence_threshold': 0.3,
                'batch_size': 8,
                'max_text_length': 512
            },
            'ml_settings': {
                'test_size': 0.2,
                'cv_folds': 5,
                'random_state': 42
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults
                config = {**default_config, **loaded_config}
            else:
                config = default_config
            
            return config
            
        except Exception as e:
            st.warning(f"Error loading config: {str(e)}. Using defaults.")
            return default_config
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str = "config.json") -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving config: {str(e)}")
            return False

class MetricsCalculator:
    """Utility class for calculating various metrics and statistics."""
    
    @staticmethod
    def calculate_text_complexity(text: str) -> Dict[str, float]:
        """
        Calculate text complexity metrics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with complexity metrics
        """
        if not text:
            return {'readability': 0, 'avg_word_length': 0, 'sentence_count': 0}
        
        # Basic text statistics
        words = text.split()
        sentences = text.count('.') + text.count('!') + text.count('?')
        sentences = max(sentences, 1)  # Avoid division by zero
        
        # Average word length
        avg_word_length = np.mean([len(word.strip(string.punctuation)) for word in words]) if words else 0
        
        # Flesch Reading Ease (simplified)
        avg_sentence_length = len(words) / sentences
        syllable_count = sum(MetricsCalculator._count_syllables(word) for word in words)
        avg_syllables_per_word = syllable_count / len(words) if words else 0
        
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        return {
            'readability': max(0, min(100, flesch_score)),
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'sentence_count': sentences,
            'word_count': len(words)
        }
    
    @staticmethod
    def _count_syllables(word: str) -> int:
        """
        Estimate syllable count in a word.
        
        Args:
            word: Input word
            
        Returns:
            Estimated syllable count
        """
        word = word.lower().strip(string.punctuation)
        if not word:
            return 0
        
        # Count vowel groups
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e'):
            syllable_count -= 1
        
        # Ensure at least one syllable
        return max(1, syllable_count)
    
    @staticmethod
    def calculate_data_quality_score(df: pd.DataFrame) -> float:
        """
        Calculate overall data quality score for a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Quality score between 0 and 1
        """
        if df.empty:
            return 0.0
        
        scores = []
        
        # Completeness score
        completeness = 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
        scores.append(completeness)
        
        # Uniqueness score (for PMID if available)
        if 'pmid' in df.columns:
            uniqueness = 1 - (df['pmid'].duplicated().sum() / len(df))
            scores.append(uniqueness)
        
        # Text quality score (for abstract if available)
        if 'abstract' in df.columns:
            avg_length = df['abstract'].str.len().mean()
            length_score = min(1.0, avg_length / 200)  # Normalize to 200 chars
            scores.append(length_score)
        
        return np.mean(scores)

def format_time_duration(seconds: float) -> str:
    """
    Format time duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def create_download_link(data: Any, filename: str, link_text: str) -> str:
    """
    Create a download link for data.
    
    Args:
        data: Data to download
        filename: Filename for download
        link_text: Text for the download link
        
    Returns:
        HTML for download link
    """
    if isinstance(data, pd.DataFrame):
        csv_data = data.to_csv(index=False)
        return f'<a href="data:text/csv;charset=utf-8,{csv_data}" download="{filename}">{link_text}</a>'
    elif isinstance(data, dict):
        json_data = json.dumps(data, indent=2)
        return f'<a href="data:application/json;charset=utf-8,{json_data}" download="{filename}">{link_text}</a>'
    else:
        return f"<span>Download not available for this data type</span>"

def log_error(error_message: str, context: str = None):
    """
    Log error message with timestamp and context.
    
    Args:
        error_message: Error message to log
        context: Optional context information
    """
    timestamp = datetime.now().isoformat()
    log_entry = f"[{timestamp}] ERROR: {error_message}"
    
    if context:
        log_entry += f" (Context: {context})"
    
    # In a real application, this would write to a log file
    # For now, we'll use Streamlit's error display
    st.error(log_entry)
