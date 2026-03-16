
# 🧠 Smart Product Pricing Challenge (Amazon ML 2025)

**Ranked Top 0.5% (121 / 22,000 teams) on Public Leaderboard**  
**Final Rank: 670 / 22,000 (Top 3%)**

---

## 📘 Project Overview
This repository contains my solution for the **Smart Product Pricing Challenge 2025**, an ML competition focused on predicting optimal e-commerce product prices from product descriptions and images.

The goal was to build a **multi-modal ML model** combining:
- 📝 **Text features** from catalog titles & descriptions
- 🧮 **Numeric features** like pack quantity, weights, and product metrics
- 🖼️ **Image embeddings** extracted via EfficientNet

---

## 🧩 Data Overview
- **train.csv:** 75k samples with product details and prices  
- **test.csv:** 75k samples without prices  
- **Fields:** `sample_id`, `catalog_content`, `image_link`, `price`

---

## ⚙️ Approach
1. **Text Preprocessing:** Regex cleaning, stopword removal, TF-IDF (5000 features)
2. **Numeric Features:** Quantity, text length, and custom engineered “fancy” features
3. **Image Features:** CNN embeddings using pretrained EfficientNet-B0
4. **Model:** LightGBM with 5-fold CV and SMAPE evaluation
5. **Optimization:** Hyperparameter tuning via Optuna, log-transformed targets

---

## 🧮 Evaluation Metric
Symmetric Mean Absolute Percentage Error (SMAPE):

\[
SMAPE = \frac{1}{n}\sum \frac{|y_{pred} - y_{true}|}{(|y_{pred}| + |y_{true}|)/2}
\]

---

## 🧾 Results
| Metric | Score |
|--------|-------|
| CV SMAPE | 12.31% |
| Public LB | **Rank 121 / 22,000** |
| Private LB | **Rank 670 / 22,000** |

---

## 📊 Feature Importance
![Feature Importance](outputs/feature_importance.png)

---

## 📦 Requirements
```bash
pip install -r requirements.txt
```
## 🔁 How to Run
1. Clone repo: `git clone ...`
2. Install dependencies: `pip install -r requirements.txt`
3. Download dataset from competition.
4. Run script `price_prediction_pipeline.py`

---

## 👥 Contributors
- [guptaayush006](https://github.com/guptaayush006)
