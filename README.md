# 💰 Prescriptive Analytics for Collections Optimization

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![Optimization](https://img.shields.io/badge/Optimization-Linear%20Programming-success)

> **Business Impact:** Demonstrated a **27.1% uplift** in daily recoverable revenue by shifting from a traditional "Risk-Based" strategy to a "Value-Based" Prescriptive Analytics engine.

## 📌 Executive Summary
In financial collections, call center capacity is a hard constraint (e.g., 50 calls/day). Traditional strategies prioritize accounts with the highest **Probability of Default (PD)**. This approach often leads to inefficient resource allocation—agents spend time chasing low-balance accounts simply because they are "risky."

This project implements a **Prescriptive Analytics Engine** that optimizes resource allocation. By combining **Machine Learning** (XGBoost/Random Forest for risk) with **Linear Programming** (PuLP for decision optimization), the model identifies the specific set of accounts that maximizes **Expected Value** ($E = P \times V$) under strict capacity constraints.

---

## 📊 Key Results

The optimization engine was tested against two baseline strategies on a synthetic dataset of 10,000 invoices with a daily capacity of 50 calls.

| Strategy | Logic | Daily Recovery ($) | Performance vs Naive |
|----------|-------|--------------------|----------------------|
| **Random** | Random Selection | ~$7,160 | -76% |
| **Naive (Baseline)** | Highest Risk (PD) First | ~$30,700 | Baseline |
| **AI Optimization** | **Maximize Expected Value** | **~$39,000** | **+27.1% 🚀** |

### Visualizing the Strategy Shift
The scatter plot below reveals *why* the AI wins. 
* **Red Dots (Naive Strategy):** Cluster on the far right (High Risk), often ignoring the balance size.
* **Green X's (AI Strategy):** Identify **"Whales"**—accounts with moderate risk but high outstanding balances—that the Naive strategy missed.

![Strategy Comparison](images/scatter_plot.png)
*(X-Axis: Probability of Default | Y-Axis: Outstanding Amount)*

---

## 🛠️ Technical Approach

### 1. Data Generation (`generate_data.py`)
Since real-world collections data is sensitive, I generated a synthetic dataset that preserves realistic statistical properties:
* **Income:** Modeled with a **Gamma Distribution** (right-skewed) to match real-world wealth distribution.
* **Risk Logic:** Constructed a latent "True PD" based on `Days_Past_Due`, `Debt_Ratio`, and `Income`, then introduced stochastic noise to simulate irreducible error.

### 2. Risk Modeling (`train_model.py`)
* **Model:** Random Forest Classifier & Regressor.
* **Calibration:** Applied **Isotonic Regression** (`CalibratedClassifierCV`) to transform raw model scores into true probabilities. This is crucial for accurate Expected Value calculations.
* **Evaluation:** ROC-AUC for discrimination; Brier Score for calibration accuracy.

### 3. Optimization Engine (`optimize.py`)
* **Framework:** Linear Programming (MILP) using the **PuLP** library.
* **Objective Function:** Maximize $\sum (Amount_i \times PD_i \times LGD_i \times InterventionEffect)$
* **Constraints:**
    * $\sum x_i \le Capacity$ (Daily limit of 50 calls).
    * $x_i \in \{0, 1\}$ (Binary decision variable).

### 4. Interactive Dashboard (`app.py`)
Built a **Streamlit** application to allow stakeholders to simulate different scenarios (e.g., changing call capacity or intervention success rates) and instantly see the financial impact.

![Dashboard Preview](images/dashboard.png)

---

## 💻 How to Run This Project

### Prerequisites
```bash
pip install -r requirements.txt
1. Generate Data
Create the synthetic dataset:

Bash
python generate_data.py
2. Train Models
Train the Risk (PD) and Severity (LGD) models:

Bash
python train_model.py
3. Run Optimization
Execute the Linear Programming solver to see the results in your terminal:

Bash
python optimize.py
4. Launch Dashboard
Start the interactive web app:

Bash
streamlit run app.py
📁 Repository Structure
src/: Source code for models and solvers.

data/: Generated datasets (collections_data.csv, modeled_data.csv).

app.py: Streamlit dashboard entry point.

optimize.py: Standalone optimization script.

generate_data.py: Synthetic data generation script.
### Prerequisites
```bash
pip install -r requirements.txt
