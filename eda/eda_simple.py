#!/usr/bin/env python3
"""
Simplified Exploratory Data Analysis (EDA) for complete_dataset_standardized.csv
This script performs comprehensive analysis of the medical/scientific text dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_examine_data():
    """Load the dataset and perform initial examination"""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("Dataset: complete_dataset_standardized.csv")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv('../data/complete_dataset_standardized.csv', sep=';')
    
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
    
    # Simple filtering for meaningful words (without NLTK)
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs'}
    
    # Filter out common words and short words
    filtered_words = []
    for word in words:
        if word.lower() not in common_words and len(word) > 2:
            filtered_words.append(word.lower())
    
    filtered_word_counts = Counter(filtered_words)
    
    print(f"\nMost common meaningful words (top 20):")
    for word, count in filtered_word_counts.most_common(20):
        print(f"  {word}: {count}")
    
    return df

def statistical_summary(df):
    """Provide statistical summary of the dataset"""
    print("\n12. STATISTICAL SUMMARY")
    print("-" * 40)
    
    print("Numerical columns summary:")
    numerical_cols = ['title_length', 'abstract_length', 'title_word_count', 'abstract_word_count']
    print(df[numerical_cols].describe())
    
    print("\nCorrelation between text lengths:")
    correlation = df['title_length'].corr(df['abstract_length'])
    print(f"Title length vs Abstract length correlation: {correlation:.4f}")
    
    return df

def create_summary_report(df):
    """Create a comprehensive summary report"""
    print("\n13. COMPREHENSIVE SUMMARY REPORT")
    print("=" * 60)
    
    # Dataset overview
    print("\nDATASET OVERVIEW:")
    print(f"- Total records: {len(df):,}")
    print(f"- Unique PMIDs: {df['pmid'].nunique():,}")
    print(f"- Duplicate PMIDs: {len(df) - df['pmid'].nunique():,}")
    print(f"- Sources: {df['source'].nunique()}")
    print(f"- Medical groups: {df['group'].nunique()}")
    
    # Text characteristics
    print("\nTEXT CHARACTERISTICS:")
    print(f"- Average title length: {df['title_length'].mean():.1f} characters")
    print(f"- Average abstract length: {df['abstract_length'].mean():.1f} characters")
    print(f"- Average title words: {df['title_word_count'].mean():.1f}")
    print(f"- Average abstract words: {df['abstract_word_count'].mean():.1f}")
    
    # Source breakdown
    print("\nSOURCE BREAKDOWN:")
    source_breakdown = df['source'].value_counts()
    for source, count in source_breakdown.items():
        percentage = (count / len(df)) * 100
        print(f"- {source}: {count:,} records ({percentage:.1f}%)")
    
    # Top medical groups
    print("\nTOP MEDICAL GROUPS:")
    group_breakdown = df['group'].value_counts().head(10)
    for group, count in group_breakdown.items():
        percentage = (count / len(df)) * 100
        print(f"- {group}: {count:,} records ({percentage:.1f}%)")
    
    # Data quality
    print("\nDATA QUALITY:")
    missing_data = df.isnull().sum()
    print(f"- Missing titles: {missing_data['title']}")
    print(f"- Missing abstracts: {missing_data['abstract']}")
    print(f"- Missing sources: {missing_data['source']}")
    print(f"- Missing groups: {missing_data['group']}")
    print(f"- Manual annotations: {missing_data['Manual']}")
    
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
        
        # Statistical summary
        df = statistical_summary(df)
        
        # Create comprehensive summary
        df = create_summary_report(df)
        
        print("\n" + "=" * 60)
        print("EDA COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nGenerated files:")
        print("- eda_visualizations.png")
        
    except Exception as e:
        print(f"Error during EDA: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 