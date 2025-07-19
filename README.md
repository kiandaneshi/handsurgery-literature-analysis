# Hand Surgery Literature Analysis Pipeline

A comprehensive machine learning pipeline for analyzing hand surgery literature using BioBERT and predicting clinical outcomes from PubMed abstracts.

## 🎯 Overview

This project represents the largest computational analysis of hand surgery literature ever conducted, processing **43,726 abstracts** from **3,107 journals** spanning 2013-2023. Using advanced NLP and machine learning techniques, we achieved **98.0% accuracy** in predicting complication risk and **89.7% accuracy** for surgical success prediction.

## 📊 Key Results

- **Dataset**: 43,726 hand surgery abstracts from 11 years of literature
- **Processing Success**: 99.1% BioBERT processing rate
- **ML Performance**: 
  - Complication Risk Prediction: 98.0% accuracy (AUC-ROC = 0.980)
  - Surgical Success Prediction: 89.7% accuracy (AUC-ROC = 0.897)
- **Features**: 71 clinical features across 6 categories
- **Models**: Logistic Regression, Random Forest, XGBoost

## 🔬 Pipeline Architecture

### 1. Literature Retrieval
- Automated PubMed search using 35 comprehensive MeSH terms
- Rate-limited API calls with NCBI Entrez integration
- Comprehensive coverage of hand surgery subspecialties

### 2. BioBERT Processing
- Clinical entity extraction using `dmis-lab/biobert-base-cased-v1.1`
- Rule-based enhancement for domain-specific terminology
- Confidence scoring and quality assessment

### 3. Feature Engineering
- 71 structured clinical features across:
  - Demographics (12 features)
  - Injury/Procedure characteristics (18 features)
  - Surgical interventions (15 features)
  - Clinical outcomes (14 features)
  - Text-based features (8 features)
  - Quality metrics (4 features)

### 4. Machine Learning
- Multi-task prediction framework
- 5-fold cross-validation with robust evaluation
- Feature importance analysis and clinical interpretation

## 🚀 Quick Start

### Citation(s)
Daneshi, K. (2025) ‘Hand Surgery Literature Analysis: End-to-End Data Extraction and Modeling Pipeline v3’. Zenodo. doi: 10.5281/zenodo.15880448. (Harvard style)

Methods paper is currently in the works.

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/hand-surgery-literature-analysis.git
cd hand-surgery-literature-analysis
pip install -r package_requirements.txt
