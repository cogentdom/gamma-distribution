import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# Set random seeds
np.random.seed(42)
# tf.random.set_seed(42)

# Generate random samples from different distributions
gamma_sample = np.random.gamma(shape=2.0, scale=2.0, size=1000)  # Gamma distribution with shape=2, scale=2
exp_sample = np.random.exponential(scale=2.0, size=1000)  # Exponential distribution with scale=2
poisson_sample = np.random.poisson(lam=5.0, size=1000)  # Poisson distribution with mean=5

import matplotlib.pyplot as plt

# Create TensorFlow gamma distribution
tf_gamma = tfp.distributions.Gamma(concentration=2.0, rate=1/2.0)  # rate is inverse of scale
try:
    tf_gamma_sample = tf_gamma.sample(1000)
except tf.errors.InvalidArgumentError as e:
    print(f"Error sampling from gamma distribution: {e}")

# Create TensorFlow exponential distribution
tf_exp = tfp.distributions.Exponential(rate=1/2.0)  # rate is inverse of scale
tf_exp_sample = tf_exp.sample(1000)

# Create TensorFlow poisson distribution
tf_poisson = tfp.distributions.Poisson(rate=5.0)  # rate parameter is equivalent to lambda in NumPy
tf_poisson_sample = tf_poisson.sample(1000)

# Create subplots for gamma, exponential, and poisson distributions
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 15))

# Gamma plots (existing)
ax1.hist(gamma_sample, bins=30, density=True, alpha=0.7, color='blue', label='NumPy')
ax1.set_title('NumPy Gamma Distribution')
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')
ax1.legend()

ax2.hist(tf_gamma_sample.numpy(), bins=30, density=True, alpha=0.7, color='red', label='TensorFlow')
ax2.set_title('TensorFlow Gamma Distribution')
ax2.set_xlabel('Value')
ax2.set_ylabel('Density')
ax2.legend()

# For Gamma distribution
x = np.linspace(0, 15, 100)
gamma_pdf = tf_gamma.prob(x)
ax1.plot(x, gamma_pdf, 'k-', label='PDF')
ax2.plot(x, gamma_pdf, 'k-', label='PDF')

# Exponential plots (new)
ax3.hist(exp_sample, bins=30, density=True, alpha=0.7, color='blue', label='NumPy')
ax3.set_title('NumPy Exponential Distribution')
ax3.set_xlabel('Value')
ax3.set_ylabel('Density')
ax3.legend()

ax4.hist(tf_exp_sample.numpy(), bins=30, density=True, alpha=0.7, color='red', label='TensorFlow')
ax4.set_title('TensorFlow Exponential Distribution')
ax4.set_xlabel('Value')
ax4.set_ylabel('Density')
ax4.legend()

# For Exponential distribution
exp_pdf = tf_exp.prob(x)
ax3.plot(x, exp_pdf, 'k-', label='PDF')
ax4.plot(x, exp_pdf, 'k-', label='PDF')

# Poisson plots
ax5.hist(poisson_sample, bins=30, density=True, alpha=0.7, color='blue', label='NumPy')
ax5.set_title('NumPy Poisson Distribution')
ax5.set_xlabel('Value')
ax5.set_ylabel('Density')
ax5.legend()

ax6.hist(tf_poisson_sample.numpy(), bins=30, density=True, alpha=0.7, color='red', label='TensorFlow')
ax6.set_title('TensorFlow Poisson Distribution')
ax6.set_xlabel('Value')
ax6.set_ylabel('Density')
ax6.legend()

# For Poisson distribution
x_poisson = np.arange(0, 15)
poisson_pmf = tf_poisson.prob(x_poisson)
ax5.plot(x_poisson, poisson_pmf, 'k-', label='PMF')
ax6.plot(x_poisson, poisson_pmf, 'k-', label='PMF')

# Print statistical comparisons
print("Gamma Distribution:")
print(f"NumPy - Mean: {np.mean(gamma_sample):.2f}, Var: {np.var(gamma_sample):.2f}")
print(f"TF    - Mean: {tf.reduce_mean(tf_gamma_sample):.2f}, Var: {tf.math.reduce_variance(tf_gamma_sample):.2f}")
print(f"Theory- Mean: {2.0 * 2.0:.2f}, Var: {2.0 * (2.0**2):.2f}")

# Calculate KL divergence between NumPy and TF samples
def calculate_kl_divergence(p, q, bins=30):
    hist_p, _ = np.histogram(p, bins=bins, density=True)
    hist_q, _ = np.histogram(q, bins=bins, density=True)
    # Add small constant to avoid division by zero
    hist_p = hist_p + 1e-10
    hist_q = hist_q + 1e-10
    return np.sum(hist_p * np.log(hist_p / hist_q))

kl_gamma = calculate_kl_divergence(gamma_sample, tf_gamma_sample)
print(f"KL Divergence (Gamma): {kl_gamma:.4f}")

plt.tight_layout()
plt.show()
