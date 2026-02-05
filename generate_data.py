import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed for reproducibility
np.random.seed(42)

def generate_collections_data(n_samples=10000):
    print(f"Generating {n_samples} synthetic accounts...")
    
    # --- 1. Generate Independent Variables (The "Features") ---
    
    # ID
    ids = [f"INV_{i:05d}" for i in range(n_samples)]
    
    # Age (Normal Distribution: Bell curve centered at 35)
    age = np.random.normal(loc=35, scale=10, size=n_samples)
    age = np.clip(age, 18, 70) # Clip to realistic ages
    
    # Income (Gamma Distribution: Right-skewed, most earn less, few earn tons)
    # Shape=2, Scale=20000 gives a realistic looking income curve
    income = np.random.gamma(shape=2, scale=20000, size=n_samples) + 20000
    
    # Outstanding Amount (Exposure at Default - EAD)
    # Correlated with income (Rich people have bigger debts), but with noise
    outstanding_amount = (income * 0.1) + np.random.normal(0, 1000, n_samples)
    outstanding_amount = np.clip(outstanding_amount, 100, 50000)
    
    # Days Past Due (DPD) (Exponential: Most people are a little late, few are very late)
    days_past_due = np.random.exponential(scale=20, size=n_samples).astype(int) + 1
    
    # --- 2. Generate The "Hidden" Risk Profile (The "Signal") ---
    
    # We construct a "True Probability of Default" (True PD) based on logic.
    # Logic: Lower Income + Higher Debt + Higher DPD = Higher Risk.
    
    # Normalize variables to 0-1 scale for the formula
    norm_income = (income - income.min()) / (income.max() - income.min())
    norm_dpd = (days_past_due - days_past_due.min()) / (days_past_due.max() - days_past_due.min())
    
    # The Formula (Linear Combination)
    # High DPD adds risk (+), High Income reduces risk (-)
    logits = -2 + (4 * norm_dpd) - (2 * norm_income) 
    
    # Convert Logits to Probability (Sigmoid Function)
    true_pd = 1 / (1 + np.exp(-logits))
    
    # --- 3. Generate Target Variables (Labels) ---
    
    # Default Flag (0 or 1) - The Bernoulli Trial
    # We flip a coin for each customer weighted by their True PD.
    default_flag = np.random.binomial(n=1, p=true_pd)
    
    # Loss Given Default (LGD) - How much do we lose if they default?
    # Modeled as a Beta distribution. 
    # If they default, we usually lose 40-80% of the money.
    true_lgd = np.random.beta(a=5, b=2, size=n_samples) 
    
    # --- 4. Assemble DataFrame ---
    df = pd.DataFrame({
        'invoice_id': ids,
        'age': age.round(1),
        'annual_income': income.round(2),
        'days_past_due': days_past_due,
        'outstanding_amount': outstanding_amount.round(2),
        'true_pd': true_pd,        # HIDDEN: You wouldn't know this in real life
        'true_lgd': true_lgd,      # HIDDEN: You wouldn't know this in real life
        'default_flag': default_flag # TARGET: This is what you train on
    })
    
    return df

# Run generation
df = generate_collections_data()

# Quick Sanity Check
print("Data Overview:")
print(df.describe())

# Check Correlation (Does the math make sense?)
# We expect DPD to be positively correlated with Default
corr = df[['days_past_due', 'annual_income', 'default_flag']].corr()
print("\nCorrelations:")
print(corr)

# Save to CSV
df.to_csv('collections_data.csv', index=False)
print("\nSuccess! 'collections_data.csv' saved.")