import pandas as pd
import numpy as np
import re
from typing import List, Dict, Optional, Callable, Tuple
import streamlit as st
from datetime import datetime
import json

# Try to import transformer dependencies, fallback to rule-based processing if not available
try:
    import torch
    from transformers import AutoTokenizer, AutoModel, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("Transformers library not available. Using rule-based entity extraction only.")

class BioBERTProcessor:
    """
    Handles clinical entity recognition and relationship extraction using BioBERT.
    Processes medical abstracts to extract structured clinical information.
    """
    
    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1", 
                 confidence_threshold: float = 0.3):
        """
        Initialize BioBERT processor.
        
        Args:
            model_name: Pre-trained BioBERT model name
            confidence_threshold: Minimum confidence for entity extraction
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        
        # Clinical entity patterns for rule-based enhancement
        self.entity_patterns = self._initialize_entity_patterns()
        
        # Entity categories to extract
        self.entity_categories = {
            'demographics': ['age_group', 'gender', 'population_type'],
            'injury_procedure': ['tendon_zone', 'fracture_type', 'nerve_type', 'anatomical_location'],
            'surgical_interventions': ['grafting', 'fixation', 'nerve_decompression', 'occupational_therapy', 'splinting'],
            'clinical_outcomes': ['grip_strength', 'range_of_motion', 'pain_reduction', 'reoperation_rate', 'return_to_work', 'complications']
        }
        
        # Confidence scoring weights
        self.confidence_weights = {
            'entity_coverage': 0.3,
            'entity_confidence': 0.4,
            'text_quality': 0.2,
            'clinical_relevance': 0.1
        }
    
    def _initialize_entity_patterns(self) -> Dict[str, List[str]]:
        """
        Initialize regex patterns for clinical entity recognition.
        
        Returns:
            Dictionary of entity patterns
        """
        return {
            'age_group': [
                r'\b(pediatric|paediatric|child|children|infant|baby)\b',
                r'\b(adult|grown)\b',
                r'\b(elderly|geriatric|senior|aged)\b',
                r'\b(\d+)\s*[-–]\s*(\d+)\s*(year|yr|y)s?\s*(old|age)\b',
                r'\b(mean|average)\s*age\s*(\d+)\b'
            ],
            'gender': [
                r'\b(male|female|men|women|man|woman)\b',
                r'\b(\d+)\s*(male|female)\b',
                r'\b(gender|sex):\s*(male|female)\b'
            ],
            'tendon_zone': [
                r'\bzone\s*[I1V2-5]+\b',
                r'\b(flexor|extensor)\s*zone\s*[I1V2-5]+\b',
                r'\b(FDP|FDS|EPL|EDC)\b'
            ],
            'fracture_type': [
                r'\b(metacarpal|phalange?al|scaphoid|radius|ulna)\s*fracture\b',
                r'\b(comminuted|spiral|transverse|oblique)\s*fracture\b',
                r'\b(open|closed|compound)\s*fracture\b'
            ],
            'nerve_type': [
                r'\b(median|ulnar|radial|digital)\s*nerve\b',
                r'\b(carpal tunnel|cubital tunnel)\b',
                r'\bneuropathy\b'
            ],
            'anatomical_location': [
                r'\b(thumb|index|middle|ring|little)\s*(finger|digit)\b',
                r'\b(wrist|hand|forearm|elbow)\b',
                r'\b(proximal|distal|middle)\s*(phalanx|phalange)\b',
                r'\b(MCP|PIP|DIP)\s*(joint)?\b'
            ],
            'surgical_interventions': [
                r'\b(tendon|nerve)\s*(graft|grafting|repair|reconstruction)\b',
                r'\b(fixation|pinning|plating|screwing)\b',
                r'\b(decompression|release|neurolysis)\b',
                r'\b(splint|splinting|immobilization)\b',
                r'\b(occupational|physical)\s*therapy\b'
            ],
            'clinical_outcomes': [
                r'\bgrip\s*strength\b',
                r'\brange\s*of\s*motion\b|\bROM\b',
                r'\bpain\s*(score|level|reduction|relief)\b',
                r'\breoperation\b|\brevision\s*surgery\b',
                r'\breturn\s*to\s*work\b|\bRTW\b',
                r'\bcomplications?\b|\badverse\s*events?\b',
                r'\b(DASH|QuickDASH|PRWE)\s*score\b'
            ]
        }
    
    def load_model(self) -> bool:
        """
        Load BioBERT model and tokenizer.
        
        Returns:
            True if successful, False otherwise
        """
        if not TRANSFORMERS_AVAILABLE:
            st.warning("Transformers library not available. Using rule-based entity extraction only.")
            self.model = None
            self.tokenizer = None
            self.ner_pipeline = None
            return True
        
        try:
            st.info(f"Loading BioBERT model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Initialize NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            st.success("BioBERT model loaded successfully!")
            return True
            
        except Exception as e:
            st.error(f"Failed to load BioBERT model: {str(e)}")
            st.info("Falling back to rule-based entity extraction only.")
            self.model = None
            self.tokenizer = None
            self.ner_pipeline = None
            return True
    
    def process_abstracts(self, abstracts_df: pd.DataFrame, batch_size: int = 8,
                         progress_callback: Optional[Callable] = None) -> Optional[List[Dict]]:
        """
        Process abstracts using BioBERT for entity extraction.
        
        Args:
            abstracts_df: DataFrame with abstracts
            batch_size: Number of abstracts to process in each batch
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of processed abstract dictionaries or None if failed
        """
        if self.model is None:
            if not self.load_model():
                return None
        
        processed_abstracts = []
        total_abstracts = len(abstracts_df)
        
        for i in range(0, total_abstracts, batch_size):
            batch_end = min(i + batch_size, total_abstracts)
            batch_df = abstracts_df.iloc[i:batch_end]
            
            if progress_callback:
                progress_callback(i, total_abstracts, f"Processing batch {i//batch_size + 1}")
            
            # Process batch
            for _, row in batch_df.iterrows():
                try:
                    processed_abstract = self._process_single_abstract(row)
                    if processed_abstract:
                        processed_abstracts.append(processed_abstract)
                
                except Exception as e:
                    st.warning(f"Failed to process PMID {row.get('pmid', 'unknown')}: {str(e)}")
                    continue
        
        # Filter by confidence threshold
        high_confidence_abstracts = [
            abstract for abstract in processed_abstracts
            if abstract['confidence_score'] >= self.confidence_threshold
        ]
        
        if progress_callback:
            progress_callback(
                total_abstracts, total_abstracts,
                f"Processed {len(processed_abstracts)} abstracts, {len(high_confidence_abstracts)} high-confidence"
            )
        
        return high_confidence_abstracts
    
    def _process_single_abstract(self, row: pd.Series) -> Optional[Dict]:
        """
        Process a single abstract for entity extraction.
        
        Args:
            row: Series containing abstract data
            
        Returns:
            Dictionary with extracted entities and metadata
        """
        try:
            # Combine title and abstract for processing
            text = f"{row.get('title', '')} {row.get('abstract', '')}"
            if len(text.strip()) < 50:
                return None
            
            # Extract entities using BioBERT
            biobert_entities = self._extract_biobert_entities(text)
            
            # Enhance with rule-based extraction
            rule_based_entities = self._extract_rule_based_entities(text)
            
            # Combine and organize entities
            combined_entities = self._combine_entities(biobert_entities, rule_based_entities)
            
            # Extract relationships
            relationships = self._extract_relationships(text, combined_entities)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(text, combined_entities, biobert_entities)
            
            return {
                'pmid': row.get('pmid'),
                'title': row.get('title', ''),
                'abstract': row.get('abstract', ''),
                'entities': combined_entities,
                'relationships': relationships,
                'confidence_score': confidence_score,
                'processing_timestamp': datetime.now().isoformat(),
                'biobert_raw_entities': biobert_entities,
                'rule_based_entities': rule_based_entities
            }
            
        except Exception as e:
            st.warning(f"Error processing abstract: {str(e)}")
            return None
    
    def _extract_biobert_entities(self, text: str) -> List[Dict]:
        """
        Extract entities using BioBERT NER pipeline.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        if not TRANSFORMERS_AVAILABLE or self.ner_pipeline is None:
            # Return empty list if transformers not available
            return []
            
        try:
            # Truncate text if too long for model
            max_length = 512
            if len(text.split()) > max_length:
                text = ' '.join(text.split()[:max_length])
            
            # Run NER pipeline
            entities = self.ner_pipeline(text)
            
            # Filter and format entities
            formatted_entities = []
            for entity in entities:
                if entity['score'] > 0.5:  # Minimum confidence for BioBERT entities
                    formatted_entities.append({
                        'text': entity['word'],
                        'label': entity['entity_group'],
                        'confidence': entity['score'],
                        'start': entity.get('start', 0),
                        'end': entity.get('end', 0),
                        'source': 'biobert'
                    })
            
            return formatted_entities
            
        except Exception as e:
            st.warning(f"BioBERT entity extraction failed: {str(e)}")
            return []
    
    def _extract_rule_based_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities using rule-based patterns.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted entities by category
        """
        entities = {}
        text_lower = text.lower()
        
        for category, patterns in self.entity_patterns.items():
            category_entities = []
            
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group().strip()
                    if entity_text and entity_text not in category_entities:
                        category_entities.append(entity_text)
            
            if category_entities:
                entities[category] = category_entities
        
        return entities
    
    def _combine_entities(self, biobert_entities: List[Dict], 
                         rule_based_entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Combine BioBERT and rule-based entities into organized categories.
        
        Args:
            biobert_entities: Entities from BioBERT
            rule_based_entities: Entities from rule-based extraction
            
        Returns:
            Combined entities organized by clinical categories
        """
        combined = {}
        
        # Initialize categories
        for main_category, subcategories in self.entity_categories.items():
            combined[main_category] = []
        
        # Add rule-based entities
        for entity_type, entities in rule_based_entities.items():
            # Map entity types to main categories
            main_category = self._map_to_main_category(entity_type)
            if main_category:
                combined[main_category].extend(entities)
        
        # Add BioBERT entities
        for entity in biobert_entities:
            entity_text = entity['text']
            entity_label = entity['label']
            
            # Map BioBERT labels to our categories
            main_category = self._map_biobert_label_to_category(entity_label, entity_text)
            if main_category:
                if entity_text not in combined[main_category]:
                    combined[main_category].append(entity_text)
        
        # Remove duplicates and empty categories
        for category in combined:
            combined[category] = list(set(combined[category]))
            if not combined[category]:
                combined[category] = []
        
        return combined
    
    def _map_to_main_category(self, entity_type: str) -> Optional[str]:
        """Map specific entity types to main categories."""
        mapping = {
            'age_group': 'demographics',
            'gender': 'demographics',
            'tendon_zone': 'injury_procedure',
            'fracture_type': 'injury_procedure',
            'nerve_type': 'injury_procedure',
            'anatomical_location': 'injury_procedure',
            'surgical_interventions': 'surgical_interventions',
            'clinical_outcomes': 'clinical_outcomes'
        }
        return mapping.get(entity_type)
    
    def _map_biobert_label_to_category(self, label: str, text: str) -> Optional[str]:
        """Map BioBERT entity labels to our clinical categories."""
        label_lower = label.lower()
        text_lower = text.lower()
        
        # Age/demographic indicators
        if any(keyword in text_lower for keyword in ['age', 'year', 'male', 'female', 'patient']):
            return 'demographics'
        
        # Anatomical/injury terms
        if any(keyword in text_lower for keyword in ['zone', 'fracture', 'nerve', 'tendon', 'finger', 'hand']):
            return 'injury_procedure'
        
        # Surgical terms
        if any(keyword in text_lower for keyword in ['surgery', 'repair', 'graft', 'fixation', 'therapy']):
            return 'surgical_interventions'
        
        # Outcome measures
        if any(keyword in text_lower for keyword in ['strength', 'motion', 'pain', 'score', 'outcome']):
            return 'clinical_outcomes'
        
        return None
    
    def _extract_relationships(self, text: str, entities: Dict[str, List[str]]) -> List[str]:
        """
        Extract relationships between clinical entities.
        
        Args:
            text: Input text
            entities: Extracted entities
            
        Returns:
            List of relationship strings
        """
        relationships = []
        text_lower = text.lower()
        
        # Simple relationship patterns
        relationship_patterns = [
            (r'(\w+)\s+(improved|decreased|increased|reduced)\s+(\w+)', 'improvement'),
            (r'(\w+)\s+(associated with|correlated with|related to)\s+(\w+)', 'association'),
            (r'(\w+)\s+(caused|resulted in|led to)\s+(\w+)', 'causation'),
            (r'(\w+)\s+(compared to|versus|vs)\s+(\w+)', 'comparison')
        ]
        
        for pattern, rel_type in relationship_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                entity1, relation, entity2 = match.groups()
                relationships.append(f"{entity1} {relation} {entity2} ({rel_type})")
        
        return relationships[:10]  # Limit to top 10 relationships
    
    def _calculate_confidence_score(self, text: str, entities: Dict[str, List[str]], 
                                  biobert_entities: List[Dict]) -> float:
        """
        Calculate confidence score for the extraction.
        
        Args:
            text: Original text
            entities: Extracted entities
            biobert_entities: Raw BioBERT entities
            
        Returns:
            Confidence score between 0 and 1
        """
        scores = {}
        
        # Entity coverage score
        total_categories = len(self.entity_categories)
        covered_categories = sum(1 for category, entity_list in entities.items() if entity_list)
        scores['entity_coverage'] = covered_categories / total_categories
        
        # Entity confidence score (from BioBERT)
        if biobert_entities:
            avg_confidence = np.mean([entity['confidence'] for entity in biobert_entities])
            scores['entity_confidence'] = avg_confidence
        else:
            scores['entity_confidence'] = 0.5  # Default for rule-based only
        
        # Text quality score
        text_length = len(text.split())
        text_quality = min(1.0, text_length / 200)  # Normalize to 200 words
        scores['text_quality'] = text_quality
        
        # Clinical relevance score
        clinical_keywords = [
            'surgery', 'treatment', 'outcome', 'patient', 'study', 'result',
            'hand', 'finger', 'tendon', 'nerve', 'fracture', 'repair'
        ]
        clinical_score = sum(1 for keyword in clinical_keywords if keyword in text.lower())
        scores['clinical_relevance'] = min(1.0, clinical_score / len(clinical_keywords))
        
        # Weighted final score
        final_score = sum(
            scores[component] * self.confidence_weights[component]
            for component in scores
        )
        
        return final_score
    
    def get_processing_statistics(self, processed_data: List[Dict]) -> Dict:
        """
        Generate statistics about the processing results.
        
        Args:
            processed_data: List of processed abstracts
            
        Returns:
            Dictionary with processing statistics
        """
        if not processed_data:
            return {}
        
        # Confidence distribution
        confidence_scores = [item['confidence_score'] for item in processed_data]
        
        # Entity extraction statistics
        entity_stats = {}
        for category in self.entity_categories:
            entity_counts = [len(item['entities'].get(category, [])) for item in processed_data]
            entity_stats[category] = {
                'mean': np.mean(entity_counts),
                'std': np.std(entity_counts),
                'max': np.max(entity_counts),
                'coverage': sum(1 for count in entity_counts if count > 0) / len(entity_counts)
            }
        
        return {
            'total_processed': len(processed_data),
            'confidence_mean': np.mean(confidence_scores),
            'confidence_std': np.std(confidence_scores),
            'high_confidence_count': sum(1 for score in confidence_scores if score >= self.confidence_threshold),
            'entity_statistics': entity_stats,
            'avg_entities_per_abstract': np.mean([
                sum(len(entities) for entities in item['entities'].values())
                for item in processed_data
            ])
        }
