# Unsupervised ML Case Study: Mental Health in Tech

This project applies unsupervised machine learning techniques to analyze and cluster employee responses from a mental health survey in the tech industry. The objective is to help Human Resources (HR) departments better understand employee mental health trends and provide targeted support.

## Dataset

- **Source**: [Mental Health in Tech Survey 2016](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
- **File Used**: `mental-heath-in-tech-2016_20161114.csv`
- **Focus**: Technology-oriented employees

## Data Preprocessing

- Removed irrelevant or incomplete columns
- Filtered out self-employed respondents
- Standardized and encoded categorical data
- Imputed missing values where appropriate
- Applied feature scaling (StandardScaler)

## Dimensionality Reduction

Used **Principal Component Analysis (PCA)** to reduce dimensionality:
- Chose components that explain a high cumulative variance
- Improved performance of clustering

## Clustering Methods

- **K-Means Clustering**
  - Optimal number of clusters determined using Elbow Method and Silhouette Score
  - Found 3 distinct employee clusters


## 📈 Evaluation Metrics

- **Elbow Plot**: For optimal `k` in K-Means
- **Silhouette Score**: To assess cluster separation and cohesion

## Key Insights

- Data clusters into 3 distinct groups based on mental health-related features
- Indicates different employee profiles with varying support/resource needs
- Findings can be used by HR to:
  - Provide targeted mental health resources
  - Design tailored wellness initiatives
  - Raise awareness and improve accessibility to mental health support


## How to run the code?
- The all of the code is found in the Final.py file.
- You can clone the repository using this:
    -git clone https://github.com/ZachSimpson123/UnsupervisedMLCaseStudyMentalHealth.git
    cd UnsupervisedMLCaseStudyMentalHealth
- The following packages were used and need to be installed before running the code.
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
    

  
