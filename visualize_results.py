import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the results
df = pd.read_csv('final_results.csv')

# Filter only the columns we need for plotting
# We want to see: Probability of Default (Risk) vs. Outstanding Amount (Reward)
plot_data = df[['pred_pd', 'outstanding_amount', 'selected_optimized', 'selected_naive_risk', 'selected_random']].copy()

# Create a label column to distinguish the strategies
# Note: Some accounts might be selected by BOTH, but we want to see the differences.
plot_data['Strategy'] = 'Ignored'
plot_data.loc[plot_data['selected_naive_risk'] == 1, 'Strategy'] = 'Naive (High Risk)'
plot_data.loc[plot_data['selected_optimized'] == 1, 'Strategy'] = 'AI Optimized'

# If both selected it, let's call it "Overlap" (to see agreement)
mask_overlap = (plot_data['selected_naive_risk'] == 1) & (plot_data['selected_optimized'] == 1)
plot_data.loc[mask_overlap, 'Strategy'] = 'Overlap'

# --- CHART 1: THE SCATTER PLOT ( The "Aha!" Moment) ---
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=plot_data[plot_data['Strategy'] != 'Ignored'], # Only show selected accounts
    x='pred_pd', 
    y='outstanding_amount', 
    hue='Strategy',
    style='Strategy',
    s=100, # Dot size
    palette={'Naive (High Risk)': 'red', 'AI Optimized': 'green', 'Overlap': 'blue'},
    alpha=0.7
)

plt.title('Why AI Wins: It Targets High Value, Not Just High Risk', fontsize=16)
plt.xlabel('Probability of Default (Risk)', fontsize=12)
plt.ylabel('Outstanding Amount (Potential Reward)', fontsize=12)
plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='50% Risk Line')
plt.legend()
plt.grid(True, alpha=0.3)

# Save
plt.savefig('strategy_comparison.png')
print("Saved 'strategy_comparison.png'")
plt.show()

# --- CHART 2: THE BUSINESS IMPACT ---
results = {
    'Random': 7161,
    'Naive (High Risk)': 30705,
    'AI Optimized': 39017
}

plt.figure(figsize=(8, 6))
bars = plt.bar(results.keys(), results.values(), color=['gray', 'red', 'green'])
plt.title('Daily Recovery Value by Strategy', fontsize=16)
plt.ylabel('Expected Value ($)', fontsize=12)

# Add text labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'${height:,.0f}',
             ha='center', va='bottom', fontsize=12, weight='bold')

plt.savefig('business_impact.png')
print("Saved 'business_impact.png'")
plt.show()