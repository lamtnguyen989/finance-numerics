import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Loading the deal.ii output
data = []
with open("output/Black-Scholes-evolution.gnuplot", "r") as BS_evolution:
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


# Plotting
df_grid = df.pivot(index="t", columns="S", values="V")

# Extract grid coordinates and values
X = df_grid.columns.values
Y = df_grid.index.values
X, Y = np.meshgrid(X, Y)
Z = df_grid.values

# Set up the figure and 3D axis
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap="viridis")

# Labels and title
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="V")
ax.set_xlabel("S")
ax.set_ylabel("t")
ax.set_zlabel("V")
ax.set_title("Finite element output")
plt.tight_layout()
plt.show()