#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for complete_dataset_standardized.csv
This script performs comprehensive analysis of the medical/scientific text dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_examine_data():
    """Load the dataset and perform initial examination"""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("Dataset: complete_dataset_standardized.csv")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv('complete_dataset_standardized.csv', sep=';')
    
    print("\n1. BASIC DATASET INFORMATION")
    print("-" * 40)
    print(f"Dataset shape: {df.shape}")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    
    print("\n2. COLUMN INFORMATION")
    print("-" * 40)
    print("Columns:", list(df.columns))
    
    print("\n3. DATA TYPES")
    print("-" * 40)
    print(df.dtypes)
    
    print("\n4. FIRST FEW ROWS")
    print("-" * 40)
    print(df.head())
    
    print("\n5. MISSING VALUES")
    print("-" * 40)
    missing_values = df.isnull().sum()
    print(missing_values)
    
    print("\n6. DUPLICATE ROWS")
    print("-" * 40)
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    
    return df

def analyze_text_columns(df):
    """Analyze text columns (title, abstract)"""
    print("\n7. TEXT ANALYSIS")
    print("-" * 40)
    
    # Title analysis
    print("\nTITLE ANALYSIS:")
    df['title_length'] = df['title'].str.len()
    print(f"Average title length: {df['title_length'].mean():.2f} characters")
    print(f"Min title length: {df['title_length'].min()} characters")
    print(f"Max title length: {df['title_length'].max()} characters")
    print(f"Title length std: {df['title_length'].std():.2f}")
    
    # Abstract analysis
    print("\nABSTRACT ANALYSIS:")
    df['abstract_length'] = df['abstract'].str.len()
    print(f"Average abstract length: {df['abstract_length'].mean():.2f} characters")
    print(f"Min abstract length: {df['abstract_length'].min()} characters")
    print(f"Max abstract length: {df['abstract_length'].max()} characters")
    print(f"Abstract length std: {df['abstract_length'].std():.2f}")
    
    # Word count analysis
    df['title_word_count'] = df['title'].str.split().str.len()
    df['abstract_word_count'] = df['abstract'].str.split().str.len()
    
    print(f"\nAverage title word count: {df['title_word_count'].mean():.2f}")
    print(f"Average abstract word count: {df['abstract_word_count'].mean():.2f}")
    
    return df

def analyze_categorical_columns(df):
    """Analyze categorical columns (source, group, Manual)"""
    print("\n8. CATEGORICAL VARIABLES ANALYSIS")
    print("-" * 40)
    
    # Source analysis
    print("\nSOURCE DISTRIBUTION:")
    source_counts = df['source'].value_counts()
    print(source_counts)
    print(f"Number of unique sources: {df['source'].nunique()}")
    
    # Group analysis
    print("\nGROUP DISTRIBUTION:")
    group_counts = df['group'].value_counts()
    print(group_counts)
    print(f"Number of unique groups: {df['group'].nunique()}")
    
    # Manual column analysis
    print("\nMANUAL COLUMN ANALYSIS:")
    manual_counts = df['Manual'].value_counts()
    print(manual_counts)
    print(f"Number of unique Manual values: {df['Manual'].nunique()}")
    
    return df

def analyze_pmid_column(df):
    """Analyze the PMID column"""
    print("\n9. PMID ANALYSIS")
    print("-" * 40)
    
    print(f"Number of unique PMIDs: {df['pmid'].nunique()}")
    print(f"Total rows: {len(df)}")
    print(f"Duplicate PMIDs: {len(df) - df['pmid'].nunique()}")
    
    # Check for missing PMIDs
    missing_pmids = df['pmid'].isnull().sum()
    print(f"Missing PMIDs: {missing_pmids}")
    
    return df

def create_visualizations(df):
    """Create various visualizations for the dataset"""
    print("\n10. CREATING VISUALIZATIONS")
    print("-" * 40)
    
    # Set up the plotting area
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Exploratory Data Analysis - Medical Dataset', fontsize=16)
    
    # 1. Title length distribution
    axes[0, 0].hist(df['title_length'], bins=30, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Title Length Distribution')
    axes[0, 0].set_xlabel('Title Length (characters)')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Abstract length distribution
    axes[0, 1].hist(df['abstract_length'], bins=30, alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('Abstract Length Distribution')
    axes[0, 1].set_xlabel('Abstract Length (characters)')
    axes[0, 1].set_ylabel('Frequency')
    
    # 3. Source distribution
    source_counts = df['source'].value_counts()
    axes[0, 2].pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
    axes[0, 2].set_title('Source Distribution')
    
    # 4. Group distribution (top 10)
    group_counts = df['group'].value_counts().head(10)
    axes[1, 0].barh(range(len(group_counts)), group_counts.values)
    axes[1, 0].set_yticks(range(len(group_counts)))
    axes[1, 0].set_yticklabels(group_counts.index)
    axes[1, 0].set_title('Top 10 Groups')
    axes[1, 0].set_xlabel('Count')
    
    # 5. Title word count distribution
    axes[1, 1].hist(df['title_word_count'], bins=20, alpha=0.7, color='orange')
    axes[1, 1].set_title('Title Word Count Distribution')
    axes[1, 1].set_xlabel('Word Count')
    axes[1, 1].set_ylabel('Frequency')
    
    # 6. Abstract word count distribution
    axes[1, 2].hist(df['abstract_word_count'], bins=30, alpha=0.7, color='pink')
    axes[1, 2].set_title('Abstract Word Count Distribution')
    axes[1, 2].set_xlabel('Word Count')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved as 'eda_visualizations.png'")
    
    return df

def text_analysis(df):
    """Perform detailed text analysis"""
    print("\n11. DETAILED TEXT ANALYSIS")
    print("-" * 40)
    
    # Combine all text for analysis
    all_text = ' '.join(df['title'].astype(str) + ' ' + df['abstract'].astype(str))
    
    # Basic text statistics
    print(f"Total characters in dataset: {len(all_text)}")
    print(f"Total words in dataset: {len(all_text.split())}")
    
    # Most common words (basic analysis)
    words = re.findall(r'\b\w+\b', all_text.lower())
    word_counts = Counter(words)
    
    print(f"\nMost common words (top 20):")
    for word, count in word_counts.most_common(20):
        print(f"  {word}: {count}")
    
    # Text preprocessing for better analysis
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Filter out stop words and lemmatize
    filtered_words = []
    for word in words:
        if word.lower() not in stop_words and len(word) > 2:
            filtered_words.append(lemmatizer.lemmatize(word.lower()))
    
    filtered_word_counts = Counter(filtered_words)
    
    print(f"\nMost common meaningful words (top 20):")
    for word, count in filtered_word_counts.most_common(20):
        print(f"  {word}: {count}")
    
    return df

def generate_wordcloud(df):
    """Generate wordcloud from text data"""
    print("\n12. GENERATING WORDCLOUD")
    print("-" * 40)
    
    # Combine all text
    all_text = ' '.join(df['title'].astype(str) + ' ' + df['abstract'].astype(str))
    
    # Create wordcloud
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         max_words=100,
                         colormap='viridis').generate(all_text)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud - Most Common Terms in Dataset', fontsize=16)
    plt.tight_layout()
    plt.savefig('wordcloud.png', dpi=300, bbox_inches='tight')
    print("Wordcloud saved as 'wordcloud.png'")
    
    return df

def statistical_summary(df):
    """Provide statistical summary of the dataset"""
    print("\n13. STATISTICAL SUMMARY")
    print("-" * 40)
    
    print("Numerical columns summary:")
    numerical_cols = ['title_length', 'abstract_length', 'title_word_count', 'abstract_word_count']
    print(df[numerical_cols].describe())
    
    print("\nCorrelation between text lengths:")
    correlation = df['title_length'].corr(df['abstract_length'])
    print(f"Title length vs Abstract length correlation: {correlation:.4f}")
    
    return df

def main():
    """Main function to run the complete EDA"""
    try:
        # Load and examine data
        df = load_and_examine_data()
        
        # Analyze text columns
        df = analyze_text_columns(df)
        
        # Analyze categorical columns
        df = analyze_categorical_columns(df)
        
        # Analyze PMID column
        df = analyze_pmid_column(df)
        
        # Create visualizations
        df = create_visualizations(df)
        
        # Perform text analysis
        df = text_analysis(df)
        
        # Generate wordcloud
        df = generate_wordcloud(df)
        
        # Statistical summary
        df = statistical_summary(df)
        
        print("\n" + "=" * 60)
        print("EDA COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nGenerated files:")
        print("- eda_visualizations.png")
        print("- wordcloud.png")
        
    except Exception as e:
        print(f"Error during EDA: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 