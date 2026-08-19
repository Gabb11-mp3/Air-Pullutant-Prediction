import numpy as np
import matplotlib.pyplot as plt

# Function definitions
def func(x):
    return -0.9 * x * x + 1.7 * x + 2.5

def derivFunc(x):
    return -1.8 * x + 1.7

# Newton-Raphson method with fixed iterations
def newtonRaphson_fixed_iterations(x, max_iterations=6):
    iterations = [(x, func(x))]  # Store iterations for plotting
    for _ in range(max_iterations):
        h = func(x) / derivFunc(x)
        x = x - h
        iterations.append((x, func(x)))
    return x, iterations

# Initial guess
x0 = 0.5

# Perform Newton-Raphson with 6 iterations
root, iterations = newtonRaphson_fixed_iterations(x0, max_iterations=6)

# Plot the function
x_vals = np.linspace(-3.5, 3, 500)
y_vals = func(x_vals)

plt.figure(figsize=(10, 12))
plt.plot(x_vals, y_vals, label="f(x) = -0.9x^2 + 1.7x + 2.5", color="blue")
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.axvline(0, color="black", linestyle="--", linewidth=0.8)

# Highlight iterations
for i, (x, y) in enumerate(iterations):
    plt.scatter(x, y, color="red", label=f"Iteration {i}" if i == 0 else "")
    plt.text(x, y, f"{i}: ({x:.4f}, {y:.4f})", fontsize=8, ha="right", color="green")

# Highlight the root
plt.scatter(root, 0, color="purple", zorder=5, label=f"Root after 6 iterations: {root:.6f}")
plt.text(root, 0.2, f"Root\n({root:.6f}, 0)", fontsize=10, color="purple", ha="center")

# Labels and legend
plt.title("Newton-Raphson Method: 6 Iterations to Find Root")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()
plt.show()
