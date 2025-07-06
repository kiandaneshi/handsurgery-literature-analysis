# Figure Captions for Manuscript

## Figure 1. Publication Trends in Hand Surgery Literature (2013-2023)
*Annual distribution of hand surgery publications retrieved from PubMed database. The graph shows consistent growth in research output over the 11-year study period, with 2021 representing the peak publication year (n=4,312). Total dataset comprised 43,726 abstracts with mean annual publication of 3,975 articles. Data points represent unique abstracts meeting inclusion criteria for comprehensive literature analysis.*

## Figure 2. BioBERT Entity Extraction Performance by Clinical Category  
*Natural language processing performance metrics for clinical entity extraction across four primary categories. Left panel shows percentage coverage of abstracts containing each entity type. Right panel displays mean number of entities extracted per abstract. Demographics showed 32.5% coverage (0.54 entities/abstract), injury/procedure showed highest coverage at 57.0% (0.91 entities/abstract), surgical interventions showed 28.0% coverage (0.34 entities/abstract), and clinical outcomes showed 39.9% coverage (0.61 entities/abstract). Overall mean entity extraction was 2.39 entities per abstract.*

## Figure 3. Machine Learning Model Performance Comparison
*Comparative analysis of three machine learning algorithms for clinical prediction tasks. Bar chart displays AUC-ROC scores for complication risk prediction (blue bars) and surgical success prediction (red bars). XGBoost achieved optimal performance for both tasks (0.980 AUC-ROC for complication risk, 0.875 AUC-ROC for surgical success), followed by Random Forest and Logistic Regression. All models demonstrated clinically meaningful performance with AUC-ROC scores exceeding 0.798.*

## Figure 4. ROC Curves for Optimal Models
*Receiver Operating Characteristic (ROC) curves for best-performing XGBoost models. Left panel shows complication risk prediction with AUC-ROC = 0.980, demonstrating near-optimal discriminative ability. Right panel shows surgical success prediction with AUC-ROC = 0.875, indicating strong predictive performance. Diagonal reference lines represent random classifier performance (AUC-ROC = 0.5). Curves approaching the upper-left corner indicate superior clinical prediction capability.*

## Figure 5. Clinical Feature Importance Analysis
*Horizontal bar chart displaying the top 10 most important clinical features for outcome prediction as determined by XGBoost gain-based feature importance. Previous complications demonstrated the highest importance score (1.182), followed by historical treatment success (0.617). Grip strength measurements, clinical documentation quality, and hand-specific terminology frequency also showed significant predictive value. Feature importance scores reflect relative contribution to model prediction accuracy and clinical decision-making capability.*

---

## Table Legends

**Table 1.** Comprehensive overview of dataset characteristics and search methodology, including database specifications, temporal coverage, search strategy implementation, and final dataset composition.

**Table 2.** Detailed performance metrics for BioBERT natural language processing pipeline, including processing success rates, confidence scoring, and entity coverage across clinical domains.

**Table 3.** Feature engineering results showing transformation of extracted clinical entities into machine learning-ready features, with target variable distribution and dataset structure summary.

**Table 4.** Comparative performance analysis of three machine learning algorithms across clinical prediction tasks, including accuracy, precision-recall metrics, and cross-validation stability assessment.

**Table 5.** Ranked importance scores for clinical features contributing to optimal model performance, with feature type classification and clinical domain categorization for interpretability.