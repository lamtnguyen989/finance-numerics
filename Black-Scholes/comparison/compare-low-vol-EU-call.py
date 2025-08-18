"""
Comparing the solution in the Low-volatility (Advection-dominated) context
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sys
sys.dont_write_bytecode = True
sys.path.insert(1, "../PINNs/src/")
from PINN import BlackScholesPINN as BlackScholesPINN

dir = "../PINNs/testing/Advection-dominated/"   # Location for data
"""
PyTorch setup
"""
# Backend setup 
torch.cuda.empty_cache()
# Set seed
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_default_dtype(torch.float32)

# CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

"""
Loading output data
"""
# Loading the deal.ii output
data = []
with open(f"{dir}Black-Scholes-evolution.gnuplot", "r") as BS_evolution:
    for line in BS_evolution:
        if not line.startswith("#") and line.strip():
            pt = line.split()
            data.append([float(pt[0]), float(pt[1]), float(pt[2])])

# Load data into a DataFrame
df = pd.DataFrame(data, columns=["S", "tau", "V"])
df = df.drop_duplicates(subset=["S", "tau"], keep="first")  # Removing duplicates

# Reverse the tau = T-t back to t
T = df["tau"].max()
df["t"] = T - df["tau"]
df = df.drop(columns=["tau"])
df = df[["S" , "t", "V"]]

"""
PINN setup
"""
param = {
    "Risk-free rate": 0.1,      # r
    "Volatility"    : 0.01,     # sigma
    "Strike price"  : 100,      # K
    "Price range"   : [50,150], # [S_min, S_max]
    "Time range"    : [0,0.25]  # [t_0, T]
}

# Paramters to set up the KAN feedforward network
KAN_param = {
    "Layers"        : [2,32,32,32,32,1],
    "Grid size"     : 10,
    "Spline order"  : 5,
    "Learning rate" : 1e-3,
    "Model path"    : "../PINNs/testing/Advection-dominated/Black-scholes-EU-call-KAN-pinn.pth"
}

# Boundary conditions for the Black-Scholes model
# This will implement the European call option
def terminal_condition(S,K):
    return torch.maximum(S-K , torch.zeros_like(S))

def max_boundary(S_max, K, r, T ,t):
    return S_max - K*torch.exp(-r*(T-t))

def min_boundary(t):
    return torch.zeros_like(t)

conditions = {
    "Terminal condition": terminal_condition,
    "Max boundary"      : max_boundary,
    "Min boundary"      : min_boundary
}

BSP = BlackScholesPINN(param=param, KAN_param=KAN_param, conditions=conditions, device=device)
BSP.load_model(BSP.model_path)

"""
Grid creation and Computation
"""
# Create grid for plotting solution
df_grid = df.pivot(index="t", columns="S", values="V")
X = df_grid.columns.values
Y = df_grid.index.values
X_mesh, Y_mesh = np.meshgrid(X, Y)
Z = df_grid.values

# PINN predictions
S = torch.tensor(X_mesh.reshape(-1, 1), dtype=torch.float32, device=BSP.device)
t = torch.tensor(Y_mesh.reshape(-1, 1), dtype=torch.float32, device=BSP.device)
V_pinn = BSP.predict(S, t).cpu().numpy().reshape(X_mesh.shape)

# Analytical solution
V_analytical = BSP.analytical_solution(S, t).cpu().numpy().reshape(X_mesh.shape)

# Calculate errors
fe_error = np.abs(Z - V_analytical)
pinn_error = np.abs(V_pinn - V_analytical)

"""
Plotting
"""
# Plotting
fig = plt.figure(figsize=(20, 10))

# Finite Element Solution 
ax1 = fig.add_subplot(2, 3, 1, projection="3d")
surf1 = ax1.plot_surface(X_mesh, Y_mesh, Z, cmap="viridis")
ax1.set_xlabel("S")
ax1.set_ylabel("t")
ax1.set_zlabel("V(S,t)")
ax1.set_title("Finite Element Solution")
fig.colorbar(surf1, ax=ax1)

# PINN Solution
ax2 = fig.add_subplot(2, 3, 2, projection="3d")
surf2 = ax2.plot_surface(X_mesh, Y_mesh, V_pinn, cmap="viridis")
ax2.set_xlabel("S")
ax2.set_ylabel("t")
ax2.set_zlabel("V(S,t)")
ax2.set_title("PINN Prediction")
fig.colorbar(surf2, ax=ax2)

# Analytical solution
ax3 = fig.add_subplot(2, 3, 3, projection="3d")
surf3 = ax3.plot_surface(X_mesh, Y_mesh, V_analytical, cmap="viridis")
ax3.set_xlabel("S")
ax3.set_ylabel("t")
ax3.set_zlabel("V(S,t)")
ax3.set_title("Analytical Solution")
fig.colorbar(surf3, ax=ax3)

# FE Pointwise Error
lvl = 100
ax4 = fig.add_subplot(2, 3, 4)
contour4 = ax4.contourf(X_mesh, Y_mesh, fe_error, levels=lvl, cmap="Reds")
ax4.set_xlabel("S")
ax4.set_ylabel("t")
ax4.set_title("FE Pointwise Error vs Analytical")
fig.colorbar(contour4, ax=ax4)

# PINN Pointwise Error
ax5 = fig.add_subplot(2, 3, 5)
contour5 = ax5.contourf(X_mesh, Y_mesh, pinn_error,  levels=lvl, cmap="Reds")
ax5.set_xlabel("S")
ax5.set_ylabel("t")
ax5.set_title("PINN Pointwise Error vs Analytical")
fig.colorbar(contour5, ax=ax5)

# Comparison at t_0
t0_idx = np.argmin(Y)  # Find index for t_0
ax6 = fig.add_subplot(2, 3, 6)
ax6.plot(X, V_analytical[t0_idx, :], "b-", label="Analytical")
ax6.plot(X, V_pinn[t0_idx, :], "r--", label="PINN Prediction")
ax6.plot(X, Z[t0_idx, :], "g:", label="Finite Element")
ax6.set_xlabel("S")
ax6.set_ylabel("V(S,t)")
ax6.set_title("Comparison at t_0")
ax6.legend()
ax6.grid(True)

plt.tight_layout()
plt.show()

# Print error stats
output = "error_stats.txt"
with open(output, "w") as f:
    print("============== Error Statistics ==============", file=f)
    print("\nFinite Element Solution:", file=f)
    print(f"Max error: {np.max(fe_error):.6f}", file=f)
    print(f"Mean error: {np.mean(fe_error):.6f}", file=f)
    print(f"RMSE: {np.sqrt(np.mean(fe_error**2)):.6f}", file=f)

    print("\nPINN Solution:", file=f)
    print(f"Max error: {np.max(pinn_error):.6f}", file=f)
    print(f"Mean error: {np.mean(pinn_error):.6f}", file=f)
    print(f"RMSE: {np.sqrt(np.mean(pinn_error**2)):.6f}", file=f)

print(f"Error statistics has been wrriten to {output}")