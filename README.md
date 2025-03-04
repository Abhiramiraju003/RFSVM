# RFSVM
# Sentiment Analysis on Product Reviews using RF-SVM

## Overview
This project implements a **Random Forest-Support Vector Machine (RF-SVM)** hybrid model for **sentiment analysis** on product reviews. The model leverages the feature selection capabilities of **Random Forest (RF)** and the classification power of **Support Vector Machine (SVM)** to classify customer reviews as **positive, negative, or neutral**.

## Dataset
The dataset consists of product reviews collected from e-commerce platforms. Each review includes:
- **Review Text**: The actual customer feedback.
- **Sentiment Label**: Categorized as Positive (1), Negative (0), or Neutral (2).

### Preprocessing Steps:
1. Tokenization and text cleaning (removal of special characters, stopwords, etc.).
2. Word embedding using **TF-IDF** or **Word2Vec**.
3. Feature selection using **Random Forest** to extract the most relevant features.
4. Training **SVM** on the selected features for sentiment classification.

## Model Architecture
The RF-SVM model consists of the following steps:
1. **Feature Selection with Random Forest**:
   - Random Forest identifies the most important features from the review text.
2. **Classification with Support Vector Machine**:
   - SVM classifies the selected features into sentiment categories.

### Hyperparameters:
- **Random Forest**:
  - Number of trees: 100-500
  - Feature importance threshold: Adjustable
- **SVM**:
  - Kernel: Linear, RBF
  - C (Regularization parameter): Adjustable
  - Gamma (for RBF kernel): Adjustable

## Installation
To run this project, install the required dependencies:
```bash
pip install numpy pandas scikit-learn nltk
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/your-repo/rfsvm-sentiment-analysis.git
cd rfsvm-sentiment-analysis
```
2. Run the preprocessing script:
```bash
python preprocess.py
```
3. Train the RF-SVM model:
```bash
python train.py
```
4. Evaluate the model:
```bash
python evaluate.py
```
5. Make predictions on new reviews:
```bash
python predict.py "The product quality is excellent!"
```

## Results
- The model achieves an accuracy of **XX%** on the test set.
- Example predictions:
  - *"I love this product!" → Positive*
  - *"The quality is terrible." → Negative*

## Future Improvements
- Implementing **Deep Learning models like LSTM or Transformer** for better performance.
- Hyperparameter tuning using **Grid Search or Bayesian Optimization**.
- Expanding the dataset to include more diverse product categories.
