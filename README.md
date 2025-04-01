# Iris_Flowers_ML_Project
---

# 🌸 Iris Dataset ML Models

This repository demonstrates the application of various Machine Learning models on the classic **Iris dataset**. The goal is to classify iris flowers among three species (*Iris-setosa, Iris-versicolor,* and *Iris-virginica*) based on their sepal and petal dimensions.

---

## 📁 Files Included

- `iris.csv` – Dataset containing 150 samples with 4 features and a class label.
- `Iris_ML_Models.py` – Code applying multiple ML algorithms and evaluating their performance.
- `Iris_RF_Model.py` – Code focused on Random Forest model with feature importance and visualizations.
- `README.md` – Overview and instructions.

---

## 🧪 Models Implemented

### 📌 1. Multiple Classifiers
Implemented using scikit-learn:
- Logistic Regression
- Linear Discriminant Analysis
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier (CART)
- Gaussian Naive Bayes
- Support Vector Machine (SVM)

**Best accuracy:** `SVM` with **98.3%**

Includes:
- Data visualization: box plots, histograms, scatter plots, and 3D plots.
- Model evaluation: accuracy, confusion matrix, classification report.
- Comparison of model accuracies using boxplot.

### 📌 2. Random Forest Classifier
- Encodes class labels for processing.
- Trains a Random Forest model with 100 trees.
- Achieved **100% accuracy** on test data.
- Displays:
  - Confusion matrix with seaborn heatmap.
  - Feature importance bar plot.

---

## 📊 Visualizations

- Feature distribution histograms
- Box and whisker plots
- Scatter matrix and 3D plots
- Model comparison boxplot
- Confusion matrix heatmap (Random Forest)
- Feature importance chart

---

## 📦 Libraries Used

```python
pandas, numpy, seaborn, matplotlib
sklearn.model_selection, sklearn.metrics, sklearn.ensemble
```
---

## 📈 Results Summary

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression| 94.1%    |
| LDA                | 97.5%    |
| KNN                | 95.8%    |
| Decision Tree      | 95.8%    |
| Naive Bayes        | 95.0%    |
| SVM                | **98.3%**|
| Random Forest      | **100%** |

