"""
Value Iteration for Exercise 1 (c)
Salesman problem with two towns A and B
Parameters: c = 3, rA = 2, rB = 1, beta = 0.9
"""

import numpy as np

# Parameters
c = 3
rA = 2
rB = 1
beta = 0.9
tolerance = 1e-6
max_iterations = 1000

# Initialize value function
V_A = 0.0
V_B = 0.0

print("Value Iteration Algorithm")
print("=" * 50)
print(f"Parameters: c = {c}, rA = {rA}, rB = {rB}, beta = {beta}")
print(f"Tolerance: {tolerance}")
print("=" * 50)
print()

for iteration in range(max_iterations):
    # Store old values
    V_A_old = V_A
    V_B_old = V_B
    
    # Bellman update for state A
    # Option 1: Stay in A
    value_stay_A = rA + beta * V_A_old
    # Option 2: Move to B (pay cost c)
    value_move_A = rA + beta * (V_B_old - c)
    V_A = max(value_stay_A, value_move_A)
    
    # Bellman update for state B
    # Option 1: Stay in B
    value_stay_B = rB + beta * V_B_old
    # Option 2: Move to A (pay cost c)
    value_move_B = rB + beta * (V_A_old - c)
    V_B = max(value_stay_B, value_move_B)
    
    # Check convergence
    error = max(abs(V_A - V_A_old), abs(V_B - V_B_old))
    
    if iteration < 5 or iteration % 10 == 0:
        print(f"Iteration {iteration}: V_A = {V_A:.6f}, V_B = {V_B:.6f}, Error = {error:.6f}")
    
    if error < tolerance:
        print(f"\nConverged after {iteration + 1} iterations!")
        break

print()
print("=" * 50)
print("Final Results:")
print("=" * 50)
print(f"V_A* = {V_A:.6f}")
print(f"V_B* = {V_B:.6f}")
print()

# Determine optimal policy
print("Optimal Policy:")
print("=" * 50)
if value_stay_A >= value_move_A:
    print("In state A: STAY (optimal)")
else:
    print("In state A: MOVE to B (optimal)")

if value_stay_B >= value_move_B:
    print("In state B: STAY (optimal)")
else:
    print("In state B: MOVE to A (optimal)")
print()

# Verify with analytical solution
print("Verification:")
print("=" * 50)
print("If optimal policy is: Stay in A, Move from B to A")
print(f"V_A (analytical) = rA / (1 - beta) = {rA} / {1 - beta} = {rA / (1 - beta):.6f}")
print(f"V_B (analytical) = rB + beta * (V_A - c) = {rB} + {beta} * ({rA / (1 - beta):.6f} - {c}) = {rB + beta * (rA / (1 - beta) - c):.6f}")

