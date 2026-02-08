import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="Collections Optimization Engine", layout="wide")

# Title and Context
st.title("💰 AI-Driven Collections Optimizer")
st.markdown("""
**The Problem:** We have limited call center capacity. Who should we call to maximize revenue?
* **Traditional Method:** Call the riskiest accounts first.
* **AI Method:** Call accounts with the highest *Expected Value* (Risk × Amount × LGD).
""")

# 1. Load Data
@st.cache_data
def load_data():
    return pd.read_csv('modeled_data.csv')

df = load_data()

# 2. Sidebar Controls (The "Interactive" Part)
st.sidebar.header("⚙️ Simulation Settings")
capacity = st.sidebar.slider("Daily Call Capacity", min_value=10, max_value=200, value=50, step=10)
intervention_impact = st.sidebar.slider("Call Success Rate (Lift)", min_value=0.05, max_value=0.50, value=0.30, step=0.05)

# 3. Dynamic Calculation
# Recalculate values based on slider inputs
df['expected_loss'] = df['outstanding_amount'] * df['pred_pd'] * df['pred_lgd']
df['value_of_call'] = df['expected_loss'] * intervention_impact

# Strategy A: Naive (Sort by Risk)
top_risk = df.sort_values(by='pred_pd', ascending=False).head(capacity)
naive_recovery = top_risk['value_of_call'].sum()

# Strategy B: AI Optimized (Sort by Value)
top_value = df.sort_values(by='value_of_call', ascending=False).head(capacity)
ai_recovery = top_value['value_of_call'].sum()

# 4. The "Money Metrics"
col1, col2, col3 = st.columns(3)
col1.metric(label="Naive Strategy Recovery", value=f"${naive_recovery:,.0f}")
col2.metric(label="AI Strategy Recovery", value=f"${ai_recovery:,.0f}")
uplift = ((ai_recovery - naive_recovery) / naive_recovery) * 100
col3.metric(label="Efficiency Gain (Uplift)", value=f"{uplift:.1f}%", delta=f"{uplift:.1f}%")

# 5. Visual Proof (The Chart you just made, but dynamic)
st.subheader("Strategy Comparison: Risk vs. Reward")

# Combine selections for plotting
# We create a new 'Status' column for the dynamic plot
df['Selection'] = 'Not Selected'
df.loc[top_risk.index, 'Selection'] = 'Naive (High Risk)'
df.loc[top_value.index, 'Selection'] = 'AI Optimized'
# Handle Overlap
overlap_idx = top_risk.index.intersection(top_value.index)
df.loc[overlap_idx, 'Selection'] = 'Overlap'

# Filter for plotting (only show selected points to keep it clean)
plot_df = df[df['Selection'] != 'Not Selected']

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=plot_df,
    x='pred_pd',
    y='outstanding_amount',
    hue='Selection',
    style='Selection',
    palette={'Naive (High Risk)': 'red', 'AI Optimized': 'green', 'Overlap': 'blue'},
    s=100,
    alpha=0.7,
    ax=ax
)
ax.set_title("Why AI Wins: Targeting High Value vs. High Risk")
ax.set_xlabel("Probability of Default (PD)")
ax.set_ylabel("Outstanding Amount ($)")
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
st.pyplot(fig)

# 6. Actionable List
st.subheader("📋 Recommended Call List for Today")
st.dataframe(top_value[['invoice_id', 'outstanding_amount', 'pred_pd', 'value_of_call']].head(10))
