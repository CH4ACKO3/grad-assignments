"""
Process XGBoost CV results CSV and generate LaTeX table
"""
import pandas as pd
import numpy as np

import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'result', 'xgboost_cv_results.csv')

# Read CV results
cv_results = pd.read_csv(csv_path)

# Select key rounds: 0, 10, 20, 50, 99
key_rounds = [0, 10, 20, 50, 99]

print("CV Results Summary:")
print("=" * 80)
print(f"Total rounds: {len(cv_results)}")
print()

# Extract data for key rounds
table_data = []
for round_num in key_rounds:
    if round_num < len(cv_results):
        row = cv_results.iloc[round_num]
        train_rmse = row['train-rmse-mean']
        test_rmse = row['test-rmse-mean']
        test_rmse_std = row['test-rmse-std']
        test_mae = row['test-mae-mean']
        test_mae_std = row['test-mae-std']
        
        table_data.append({
            'round': round_num,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_rmse_std': test_rmse_std,
            'test_mae': test_mae,
            'test_mae_std': test_mae_std
        })
        
        print(f"Round {round_num}:")
        print(f"  Train RMSE: {train_rmse:.2f}")
        print(f"  Test RMSE: {test_rmse:.2f} (± {test_rmse_std:.2f})")
        print(f"  Test MAE: {test_mae:.2f} (± {test_mae_std:.2f})")
        print()

# Generate LaTeX table code
print("\n" + "=" * 80)
print("LaTeX Table Code:")
print("=" * 80)
print()
print("        Round & Train RMSE & Test RMSE & Test RMSE (std) & Test MAE & Test MAE (std) \\\\")
print("        \\midrule")
for data in table_data:
    print(f"        {data['round']} & {data['train_rmse']:.2f} & {data['test_rmse']:.2f} & {data['test_rmse_std']:.2f} & {data['test_mae']:.2f} & {data['test_mae_std']:.2f} \\\\")
print("        \\bottomrule")

# Also print final results
print()
print("=" * 80)
print("Final Results (Round 99):")
print("=" * 80)
final_row = cv_results.iloc[-1]
print(f"Final CV RMSE: {final_row['test-rmse-mean']:.4f} (± {final_row['test-rmse-std']:.4f})")
print(f"Final CV MAE: {final_row['test-mae-mean']:.4f} (± {final_row['test-mae-std']:.4f})")

