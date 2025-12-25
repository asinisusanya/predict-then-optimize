![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Operations Research](https://img.shields.io/badge/Operations%20Research-Optimization-green)
![MILP](https://img.shields.io/badge/MILP-PuLP%20%2B%20CBC-red)

# Predict–Then–Optimize Framework for Nurse Staffing

This project investigates how demand prediction errors affect nurse staffing decisions and operational cost using a **Predict–Then–Optimize (PTO)** framework.  
It combines **machine learning demand forecasting** with **mixed-integer optimization** to demonstrate that **forecast accuracy alone is insufficient for decision-making** in hospital staffing systems.

---

## 🔍 Problem Statement

Hospitals must decide how many nurses to assign to each shift **before knowing true future demand**.  
In practice, this is done by:
1. Predicting demand using historical data
2. Optimizing staffing based on the prediction

Most existing approaches evaluate demand forecasts **only using accuracy metrics** (MAE, RMSE).  
However, accurate predictions do not necessarily lead to **optimal staffing decisions**.

**Key Question:**  
> How do forecast errors, noise, and demand drift affect staffing decisions and cost when predictions are used inside an optimization model?

---

## 🧠 Core Insight

> **Prediction accuracy ≠ Decision quality**

The project demonstrates that:
- Forecasts with higher error can appear cheaper due to under-prediction
- Random noise can hide operational risk
- Systematic demand drift causes persistent staffing bias and higher cost
- Night shifts amplify forecast errors disproportionately

---

## ⚙️ Methodology (Predict–Then–Optimize)

### 1. Synthetic Data Generation
- Simulates one hospital ward over 12 months
- Three shifts per day: Morning, Evening, Night
- Includes hidden high-demand (casualty) days
- Ground truth demand is known but not available to ML models

### 2. Demand Forecasting (ML Layer)
- Models: Linear Regression, Random Forest
- Features: day-of-week, lagged demand, shift encoding
- Casualty indicators are not used
- Evaluation using MAE and RMSE (diagnostic only)

### 3. Staffing Optimization (OR Layer)
- Mixed-Integer Linear Program (MILP)
- Objective: minimize wage cost + understaffing penalties
- Constraints:
  - Shift-based capacity limits
  - Integer staffing decisions
  - 95% service-level requirement
- Implemented using **PuLP + CBC solver**

### 4. Experimental Scenarios
The same optimization model is solved under different demand inputs:
- Perfect (oracle) forecast
- Machine learning forecast
- Noisy forecast
- Drifted forecast

---

## 📊 Key Results

- ML and noisy forecasts result in lower apparent cost due to demand smoothing
- Drifted forecasts lead to higher cost than the perfect forecast
- Forecast direction and structure matter more than error magnitude
- Decision-aware evaluation reveals risks hidden by accuracy metrics

---

## 📁 Repository Structure

```
predict-then-optimize/
│
├── data/ -> Raw and processed datasets
├── src/ -> Core implementation (generation, forecasting, optimization)
├── notebooks/ -> EDA, analysis, visualization
├── results/ -> Predictions, figures, summary tables
├── report/ -> LaTeX report source files
├── README.md
├── requirements.txt
└── .gitignore
```

## 👤 Author  
Asini Susanya
