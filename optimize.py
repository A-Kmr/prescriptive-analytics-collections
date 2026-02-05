import pandas as pd
import pulp

# 1. Load the Modeled Data
df = pd.read_csv('modeled_data.csv')

# Let's verify we have the predictions
print(f"Loaded {len(df)} invoices.")

# --- 2. DEFINE THE PARAMETERS ---
CALL_CAPACITY = 50  # We can only make 50 calls today
INTERVENTION_EFFECT = 0.30 # A call saves 30% of the Expected Loss (Assumption)

# Calculate the "Value" of calling each person
# Value = How much money we save if we make the call
df['expected_loss'] = df['outstanding_amount'] * df['pred_pd'] * df['pred_lgd']
df['value_of_call'] = df['expected_loss'] * INTERVENTION_EFFECT

# --- 3. BUILD THE OPTIMIZER (Linear Programming) ---
# Initialize the "Problem" - We want to MAXIMIZE value
prob = pulp.LpProblem("Collections_Optimization", pulp.LpMaximize)

# Create Decision Variables (Binary: 1 = Call, 0 = Don't Call)
# We create a dictionary of variables, one for each row index
invoice_indices = df.index.tolist()
decision_vars = pulp.LpVariable.dicts("Call", invoice_indices, cat='Binary')

# OBJECTIVE FUNCTION: Maximize the Sum of 'Value of Call' for selected invoices
prob += pulp.lpSum([df['value_of_call'][i] * decision_vars[i] for i in invoice_indices])

# CONSTRAINT: Total calls must be <= Capacity
prob += pulp.lpSum([decision_vars[i] for i in invoice_indices]) <= CALL_CAPACITY

# --- 4. SOLVE ---
print(f"Optimizing selection for {CALL_CAPACITY} slots...")
prob.solve()

# Check Status (Optimal means it found the best solution)
print(f"Status: {pulp.LpStatus[prob.status]}")

# --- 5. EXTRACT RESULTS ---
# Read the decision variables back into the DataFrame
df['selected_optimized'] = [int(decision_vars[i].varValue) for i in invoice_indices]

# --- 6. PROVE THE VALUE (The "Money Slide") ---

# Strategy A: Random Selection (The Baseline)
df['selected_random'] = 0
random_indices = df.sample(n=CALL_CAPACITY, random_state=42).index
df.loc[random_indices, 'selected_random'] = 1

# Strategy B: Naive "High Risk" Selection (Call highest PD)
df['selected_naive_risk'] = 0
top_risk_indices = df.sort_values(by='pred_pd', ascending=False).head(CALL_CAPACITY).index
df.loc[top_risk_indices, 'selected_naive_risk'] = 1

# Calculate Total Value for each strategy
value_optimized = df[df['selected_optimized'] == 1]['value_of_call'].sum()
value_random = df[df['selected_random'] == 1]['value_of_call'].sum()
value_naive = df[df['selected_naive_risk'] == 1]['value_of_call'].sum()

print("\n--- RESULTS (Money Saved per Day) ---")
print(f"Random Selection:   ${value_random:,.2f}")
print(f"Naive (Highest PD): ${value_naive:,.2f}")
print(f"Optimized (AI):     ${value_optimized:,.2f}")

uplift = ((value_optimized - value_naive) / value_naive) * 100
print(f"IMPACT: The AI Optimizer saved {uplift:.1f}% more money than the Naive strategy.")

# Save final results
df.to_csv('final_results.csv', index=False)