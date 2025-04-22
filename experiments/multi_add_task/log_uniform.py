import numpy as np

def generate_M_for_log_uniform_tau(shape, tau_min=0.1, tau_max=100.0):
    """
    Generates M such that tau = -1 / log(sigmoid(M)) is log-uniform in [tau_min, tau_max].
    
    Args:
        shape: Shape of the output matrix (e.g., (100, 100)).
        tau_min: Minimum tau value (must be > 0).
        tau_max: Maximum tau value (must be > tau_min).
    
    Returns:
        M: Matrix where tau derived from sigmoid(M) is log-uniform.
    """
    # Sample U uniformly in log space
    log_tau_min = np.log(tau_min)
    log_tau_max = np.log(tau_max)
    U = np.random.uniform(log_tau_min, log_tau_max, size=shape)
    
    # Compute M = -log(exp(exp(-U)) - 1)
    M = -np.log(np.exp(np.exp(-U)) - 1)
    
    return M

# Example usage
M = generate_M_for_log_uniform_tau((10000,), tau_min=1.0, tau_max=500.0)
alpha = 1 / (1 + np.exp(-M))  # sigmoid(M)
tau = -1 / np.log(alpha)       # Derived tau

# Plot histogram of log(tau) to verify uniformity
import matplotlib.pyplot as plt
plt.hist(np.log(tau), bins=50, density=True)
plt.title("log(tau) ~ Uniform (log-uniform tau)")
plt.xlabel("log(tau)")
plt.ylabel("Density")
plt.show()

print(np.sum(tau<10)/10000.0)
print(np.sum(tau>100)/10000.0)

plt.hist(tau, bins=50, density=True)
plt.title("tau ~ Uniform (log-uniform tau)")
plt.xlabel("log(tau)")
plt.ylabel("Density")
plt.show()