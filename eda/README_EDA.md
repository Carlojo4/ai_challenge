# Exploratory Data Analysis (EDA) Report
## Dataset: complete_dataset_standardized.csv

### Overview
This document provides a comprehensive analysis of the medical/scientific text dataset containing 4,195 records with 6 columns.

### Dataset Structure

#### Columns:
- **pmid**: PubMed ID (integer)
- **title**: Article title (text)
- **abstract**: Article abstract (text)
- **source**: Data source (categorical)
- **group**: Medical classification (categorical)
- **Manual**: Manual annotation flag (mostly null)

### Key Findings

#### 1. Dataset Size and Quality
- **Total Records**: 4,195
- **Unique PMIDs**: 3,006 (1,189 duplicate PMIDs)
- **Missing Data**: Only the 'Manual' column has missing values (4,187 out of 4,195)
- **Data Completeness**: Excellent - all core fields (pmid, title, abstract, source, group) are complete

#### 2. Data Sources Distribution
- **Synthetic Data**: 2,382 records (56.8%) - Largest source
- **BC5CDR**: 1,275 records (30.4%) - Second largest
- **NCBI**: 538 records (12.8%) - Smallest source

#### 3. Medical Classification Groups
The dataset covers 15 different medical groups, with the following top categories:

1. **Neurological**: 1,244 records (29.7%)
2. **Cardiovascular**: 758 records (18.1%)
3. **Hepatorenal**: 628 records (15.0%)
4. **Neurological|Cardiovascular**: 363 records (8.7%)
5. **Oncological**: 279 records (6.7%)

#### 4. Text Characteristics

##### Titles:
- **Average Length**: 69.5 characters
- **Average Word Count**: 8.8 words
- **Range**: 20-294 characters
- **Distribution**: Right-skewed with most titles being concise

##### Abstracts:
- **Average Length**: 696.3 characters
- **Average Word Count**: 100.0 words
- **Range**: 146-3,814 characters
- **Distribution**: Highly variable with some very long abstracts

#### 5. Text Analysis Insights

##### Most Common Medical Terms:
1. **patients**: 5,307 occurrences
2. **results**: 3,109 occurrences
3. **cancer**: 2,433 occurrences
4. **study**: 2,298 occurrences
5. **methods**: 2,262 occurrences
6. **disease**: 2,053 occurrences
7. **conclusion**: 1,823 occurrences
8. **induced**: 1,700 occurrences
9. **treatment**: 1,414 occurrences
10. **patient**: 1,182 occurrences

#### 6. Statistical Correlations
- **Title vs Abstract Length**: Strong positive correlation (0.7251)
- This suggests that longer titles tend to have longer abstracts

### Data Quality Assessment

#### Strengths:
- ✅ Complete data for core fields
- ✅ No duplicate rows
- ✅ Consistent data types
- ✅ Rich medical terminology
- ✅ Good coverage across medical domains

#### Areas of Note:
- ⚠️ 1,189 duplicate PMIDs (28.3% of records)
- ⚠️ Manual annotation column mostly empty (99.8% missing)
- ⚠️ High variability in abstract lengths

### Medical Domain Coverage

The dataset provides comprehensive coverage across multiple medical domains:

1. **Neurological Disorders** (29.7%): Largest category, covering brain and nervous system conditions
2. **Cardiovascular Diseases** (18.1%): Heart and circulatory system conditions
3. **Hepatorenal Conditions** (15.0%): Liver and kidney related disorders
4. **Oncological** (6.7%): Cancer-related research
5. **Multi-domain combinations**: Various combinations of the above domains

### Recommendations for Use

#### For Machine Learning:
- **Text Classification**: Suitable for medical text classification tasks
- **Multi-label Classification**: Groups can be split by '|' for multi-label learning
- **Text Generation**: Rich abstracts for medical text generation models
- **Information Extraction**: Good for NER and relation extraction tasks

#### Data Preprocessing Considerations:
- Handle duplicate PMIDs appropriately
- Consider text length normalization
- Implement domain-specific preprocessing for medical terms
- Use source information for stratified sampling

#### Potential Applications:
1. **Medical Text Classification**: Classify articles into medical domains
2. **Abstract Summarization**: Generate concise summaries from long abstracts
3. **Medical Information Retrieval**: Build search systems for medical literature
4. **Multi-label Learning**: Predict multiple medical domains per article

### Files Generated
- `eda_visualizations.png`: Comprehensive visualization dashboard
- `eda_simple.py`: Complete EDA script
- `requirements_eda.txt`: Dependencies for the analysis

### Technical Notes
- Dataset uses semicolon (;) as delimiter
- Text encoding appears to be UTF-8
- No special characters or encoding issues detected
- All text is in English

---

*Analysis performed on: complete_dataset_standardized.csv*
*Total analysis time: < 1 minute*
*Generated visualizations: 6 comprehensive charts* 