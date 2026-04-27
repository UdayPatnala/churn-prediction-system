import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 250

data = {
    'customer_id': [f"CU{str(i).zfill(3)}" for i in range(1, n_samples + 1)],
    'tenure': np.random.randint(1, 72, n_samples),
    'monthly_charges': np.random.uniform(20, 120, n_samples).round(2),
    'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.25, 0.25]),
    'has_internet': np.random.choice(['Yes', 'No'], n_samples, p=[0.8, 0.2]),
    'has_phone': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
    'support_tickets': np.random.randint(0, 10, n_samples),
}

df = pd.DataFrame(data)

# Derive total_charges
df['total_charges'] = (df['tenure'] * df['monthly_charges']).round(2)

# Create synthetic target 'churn' with some logical correlations
churn_prob = np.zeros(n_samples)
churn_prob += np.where(df['contract'] == 'Month-to-month', 0.3, 0.05)
churn_prob += np.where(df['tenure'] < 12, 0.2, 0.0)
churn_prob += np.where(df['support_tickets'] > 4, 0.3, 0.0)
churn_prob += np.where(df['monthly_charges'] > 80, 0.1, 0.0)

# Normalize and sample
churn_prob = np.clip(churn_prob, 0, 1)
df['churn'] = np.random.binomial(1, churn_prob)

df.to_csv('data/customer_churn_sample.csv', index=False)
print("Expanded dataset to 250 records.")
