import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# Generate random samples from different distributions
gamma_sample = np.random.gamma(shape=2.0, scale=2.0, size=1000)  # Gamma distribution with shape=2, scale=2
exp_sample = np.random.exponential(scale=2.0, size=1000)  # Exponential distribution with scale=2
poisson_sample = np.random.poisson(lam=5.0, size=1000)  # Poisson distribution with mean=5

import matplotlib.pyplot as plt

# Create TensorFlow gamma distribution
tf_gamma = tfp.distributions.Gamma(concentration=2.0, rate=1/2.0)  # rate is inverse of scale
tf_gamma_sample = tf_gamma.sample(1000)

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

ax2.hist(tf_gamma_sample, bins=30, density=True, alpha=0.7, color='red', label='TensorFlow')
ax2.set_title('TensorFlow Gamma Distribution')
ax2.set_xlabel('Value')
ax2.set_ylabel('Density')
ax2.legend()

# Exponential plots (new)
ax3.hist(exp_sample, bins=30, density=True, alpha=0.7, color='blue', label='NumPy')
ax3.set_title('NumPy Exponential Distribution')
ax3.set_xlabel('Value')
ax3.set_ylabel('Density')
ax3.legend()

ax4.hist(tf_exp_sample, bins=30, density=True, alpha=0.7, color='red', label='TensorFlow')
ax4.set_title('TensorFlow Exponential Distribution')
ax4.set_xlabel('Value')
ax4.set_ylabel('Density')
ax4.legend()

# Poisson plots
ax5.hist(poisson_sample, bins=30, density=True, alpha=0.7, color='blue', label='NumPy')
ax5.set_title('NumPy Poisson Distribution')
ax5.set_xlabel('Value')
ax5.set_ylabel('Density')
ax5.legend()

ax6.hist(tf_poisson_sample, bins=30, density=True, alpha=0.7, color='red', label='TensorFlow')
ax6.set_title('TensorFlow Poisson Distribution')
ax6.set_xlabel('Value')
ax6.set_ylabel('Density')
ax6.legend()

plt.tight_layout()
plt.show()
