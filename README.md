# 🌲 Forest Cover Type Classification  

This project is part of the **Elevvo Pathways Machine Learning Internship**. It uses the **Covertype dataset** to classify different forest cover types based on cartographic features.  

---

## 🚀 Features
- Trains and evaluates two models:  
  - **Random Forest** 🌲  
  - **XGBoost** 🚀  
- Compares accuracy and classification metrics.  
- Generates:  
  - Confusion matrices  
  - Top feature importance plots  

---

## ⚙️ Tech Stack
- **Python**  
- **Pandas, NumPy** (data preprocessing)  
- **Scikit-learn** (Random Forest, metrics)  
- **XGBoost** (gradient boosting model)  
- **Matplotlib, Seaborn** (plots & visualizations)  

---

## 📂 Dataset
Uses the **Covertype dataset**:  
- 54 cartographic features  
- Target: 7 forest cover types  

Available from [UCI Repository](https://archive.ics.uci.edu/ml/datasets/covertype).  

---

## ▶️ Run the Project
```bash
# Clone repo
git clone https://github.com/yourusername/covtype-classification.git
cd covtype-classification

# Install requirements
pip install -r requirements.txt

# Run script
python covtype.py
📊 Output
Console: accuracy, precision, recall, F1 scores for each class.

PNGs saved:

rf_confusion.png

rf_feature_importance.png

xgb_confusion.png

xgb_feature_importance.png

