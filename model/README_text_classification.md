# Text Classification for Scientific Articles

This notebook implements a comprehensive text classification system for categorizing scientific articles based on their title and abstract content.

## Overview

The notebook follows best practices from the literature for text classification tasks and includes:

### 1. Data Exploration and Analysis
- **Dataset loading and inspection**: Analysis of the challenge_data.csv file
- **Text statistics**: Character and word count distributions
- **N-gram analysis**: Unigrams and bigrams frequency analysis
- **Class distribution**: Visualization of article group categories

### 2. Text Preprocessing
- **Custom stopwords**: Scientific domain-specific stopwords to improve classification
- **Text cleaning**: Punctuation removal, lowercase conversion, whitespace normalization
- **Tokenization and lemmatization**: Using NLTK for advanced text processing
- **Stopwords customization**: Extended stopwords dictionary for scientific articles

### 3. Feature Engineering
- **TF-IDF vectorization**: With configurable n-gram ranges (1-2)
- **Feature selection**: Configurable min_df and max_df parameters
- **Vocabulary optimization**: Maximum features limit for computational efficiency

### 4. Model Development
- **Baseline model**: Multinomial Naive Bayes as a simple baseline
- **Advanced models**: Logistic Regression, Linear SVM, Random Forest
- **Hyperparameter optimization**: Grid search with cross-validation
- **Model comparison**: Comprehensive evaluation metrics

### 5. Best Practices Implementation
- **Stratified sampling**: 85% train / 15% test split maintaining class distribution
- **Cross-validation**: 5-fold CV for robust hyperparameter selection
- **Feature importance analysis**: Understanding model decisions
- **Model persistence**: Saving best models for deployment

## Key Features

### Custom Stopwords
The notebook includes an extensive list of scientific domain-specific stopwords that commonly appear in research papers but don't contribute to classification:
- Research methodology terms (study, analysis, method, etc.)
- Common scientific phrases (results, conclusions, findings, etc.)
- Publication-related terms (paper, journal, conference, etc.)

### N-gram Analysis
- **Unigrams**: Single word frequency analysis
- **Bigrams**: Two-word phrase frequency analysis
- Helps identify domain-specific terminology and phrases

### Model Pipeline
- **Scikit-learn Pipeline**: Ensures consistent preprocessing across train/test splits
- **Hyperparameter optimization**: Systematic search for optimal parameters
- **Model evaluation**: Multiple metrics and cross-validation scores

## Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements_text_classification.txt
   ```

2. **Run the notebook**: Open `text_classification.ipynb` in Jupyter

3. **Data requirements**: Ensure `challenge_data.csv` is in the `../data/` directory

4. **Expected outputs**:
   - Comprehensive data analysis visualizations
   - Model performance comparisons
   - Best model saved as pickle file
   - Custom stopwords saved for future use

## Model Performance

The notebook evaluates multiple algorithms and provides:
- **Accuracy scores**: Both cross-validation and test set performance
- **Classification reports**: Precision, recall, F1-score per class
- **Feature importance**: Understanding what drives classification decisions
- **Model comparison**: Side-by-side performance analysis

## Recommendations for Production

1. **Ensemble methods**: Combine multiple models for improved performance
2. **Data augmentation**: Generate synthetic training examples for underrepresented classes
3. **Regular retraining**: Update models with new data to maintain performance
4. **Model monitoring**: Track performance drift over time
5. **Feature engineering**: Experiment with additional text features (POS tags, named entities)

## Technical Details

- **Python version**: 3.8+
- **Key libraries**: scikit-learn, NLTK, pandas, matplotlib, seaborn
- **Vectorization**: TF-IDF with configurable parameters
- **Cross-validation**: 5-fold stratified CV
- **Random seed**: Fixed for reproducibility

## File Structure

```
model/
├── text_classification.ipynb          # Main notebook
├── requirements_text_classification.txt # Dependencies
├── README_text_classification.md      # This file
└── [generated model files]            # Saved models after execution
```

## Notes

- The notebook is designed to be run sequentially
- All random seeds are fixed for reproducibility
- Large datasets may require significant computation time for hyperparameter optimization
- Consider using GPU acceleration for large-scale experiments 