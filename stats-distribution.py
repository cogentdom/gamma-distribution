import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import math  # Add math import
import matplotlib.pyplot as plt


# plt.style.use('ggplot')
# plt.rc('patch', force_edgecolor=True,edgecolor='black')
# plt.rc('hist', bins='auto')
# sns.set_style("darkgrid")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set Seaborn style
# sns.set_theme(style="whitegrid")
# Get the current Seaborn color palette
colors = sns.color_palette("Set2", 3).as_hex()

class DistributionComparison:
    def __init__(self, sample_size=1000):
        self.sample_size = sample_size
        
        # Distribution parameters
        self.gamma_shape = 2.0
        self.gamma_scale = 2.0
        self.exp_scale = 2.0
        self.poisson_lambda = 5.0
        
        # Generate samples (using NumPy only now)
        self.generate_samples()
        
    def generate_samples(self):
        # Generate samples using NumPy only
        self.gamma = np.random.gamma(shape=self.gamma_shape, scale=self.gamma_scale, size=self.sample_size)
        self.exp = np.random.exponential(scale=self.exp_scale, size=self.sample_size)
        self.poisson = np.random.poisson(lam=self.poisson_lambda, size=self.sample_size)
    
    def calculate_kl_divergence(self, p, q, bins=30):
        """Calculate KL divergence between two sample sets"""
        hist_p, _ = np.histogram(p, bins=bins, density=True)
        hist_q, _ = np.histogram(q, bins=bins, density=True)
        # Add small constant to avoid division by zero
        hist_p = hist_p + 1e-10
        hist_q = hist_q + 1e-10
        return np.sum(hist_p * np.log(hist_p / hist_q))
    
    def create_dashboard(self):
        """Create an interactive Plotly dashboard"""
        # Create subplots with Seaborn-inspired styling
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Empirical Distributions', 'Theoretical PDFs/PMFs',
                          'Distribution Statistics', 'KL Divergence Matrix'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                  [{"type": "table"}, {"type": "heatmap"}]],
            horizontal_spacing=0.15,
            vertical_spacing=0.17
        )
        
        # Plot 1: Empirical distributions with Seaborn colors
        fig.add_trace(
            go.Histogram(x=self.gamma, name='Gamma', opacity=0.75,
                        nbinsx=15, histnorm='probability density',
                        marker_color=colors[0],
                        showlegend=True),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=self.exp, name='Exponential', opacity=0.75,
                        nbinsx=15, histnorm='probability density',
                        marker_color=colors[1],
                        showlegend=True),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=self.poisson, name='Poisson', opacity=0.75,
                        nbinsx=15, histnorm='probability density',
                        marker_color=colors[2],
                        showlegend=True),
            row=1, col=1
        )
        
        # Plot 2: Theoretical PDFs/PMFs with Seaborn colors
        x = np.linspace(0, 15, 100)
        x_poisson = np.arange(0, 15)
        
        gamma_pdf = self._gamma_pdf(x, self.gamma_shape, self.gamma_scale)
        exp_pdf = self._exponential_pdf(x, self.exp_scale)
        poisson_pmf = self._poisson_pmf(x_poisson, self.poisson_lambda)
        
        fig.add_trace(
            go.Scatter(x=x, y=gamma_pdf, name='Gamma PDF',
                      line=dict(color=colors[0], width=3),
                      showlegend=True),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=x, y=exp_pdf, name='Exponential PDF',
                      line=dict(color=colors[1], width=3),
                      showlegend=True),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=x_poisson, y=poisson_pmf, name='Poisson PMF',
                      line=dict(color=colors[2], width=3),
                      showlegend=True),
            row=1, col=2
        )
        
        # Plot 3: Statistics Table with Seaborn-inspired colors
        stats_data = self._get_statistics_table_data()
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Distribution', 'Mean', 'Variance', 'Skewness', 'Kurtosis'],
                    # fill_color=sns.color_palette("light:blue", n_colors=1).as_hex()[0],
                    align='left',
                    height=40,
                    font=dict(size=13),
                    line=dict(width=0)
                ),
                cells=dict(
                    values=stats_data,
                    # fill_color=sns.color_palette("light:blue", n_colors=2).as_hex()[1],
                    align='left',
                    height=30
                ),
                columnwidth=[150, 100, 100, 100, 100]
            ),
            row=2, col=1
        )
        
        # Plot 4: KL Divergence Heatmap with Seaborn colors
        kl_matrix, labels = self._get_kl_divergence_matrix()
        fig.add_trace(
            go.Heatmap(
                z=kl_matrix,
                x=labels,
                y=labels,
                colorscale='Earth',
                showscale=True
            ),
            row=2, col=2
        )
        
        # Update layout with Seaborn-inspired styling
        fig.update_layout(
            height=1400,
            width=1400,
            showlegend=True,
            title_text="Distribution Analysis Dashboard",
            title_x=0.5,
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                yanchor="top",
                y=1.1,
                xanchor="center",
                x=0.5,
                orientation="h",
                bgcolor="rgba(255, 255, 255, 0.8)",
                # bordercolor=sns.color_palette("gray")[2],
                borderwidth=1,
                font=dict(size=12),
                itemsizing="constant"
            ),
            margin=dict(t=0, l=10, r=10, b=0)
        )
        
        # Update axes with Seaborn-style grid
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            # gridcolor=sns.color_palette("gray")[1],
            title_text="Value",
            title_font=dict(size=12),
            tickfont=dict(size=10)
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            # gridcolor=sns.color_palette("gray")[1],
            title_font=dict(size=12),
            tickfont=dict(size=10)
        )
        
        # Add descriptive annotations
        descriptions = [
            dict(
                x=0.02, y=0.51,  # Moved down slightly
                xref="paper", yref="paper",
                text=(
                    "<b>Distribution Characteristics:</b><br>" +
                    "• <b>Gamma (α=2, β=2):</b> Right-skewed, continuous distribution.<br> " +
                    "Shows a peak and long right tail. Used for modeling waiting times and positive-valued data.<br>" +
                    "• <b>Exponential (λ=0.5):</b> Special case of Gamma. Memoryless property, constant hazard rate.<br> " +
                    "Models time between events.<br>" +
                    "• <b>Poisson (λ=5):</b> Discrete distribution for count data. Models number of events in fixed interval."
                ),
                showarrow=False,
                font=dict(size=12),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(0, 0, 0, 0.3)",
                borderwidth=1,
                borderpad=4
            ),
            dict(
                x=0.95, y=0.52,  # Moved to right side and down
                xref="paper", yref="paper",
                text=(
                    "<b>Distribution Comparisons:</b><br>" +
                    "• <b>Type:</b> Poisson (discrete) vs Gamma/Exponential (continuous)<br>" +
                    "• <b>Shape:</b> Exponential (decreasing) vs Gamma (peaked) vs Poisson (bell)<br>" +
                    "• <b>Mean/Variance:</b> Poisson unique with equal mean and variance"
                ),
                showarrow=False,
                font=dict(size=12),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(0, 0, 0, 0.3)",
                borderwidth=1,
                borderpad=4
            ),
            dict(
                x=0.22, y=0.25,  # Adjusted for bottom left
                xref="paper", yref="paper",
                text=(
                    "<b>Statistical Measures:</b><br>" +
                    "• <b>Mean & Variance:</b> Central tendency and spread<br>" +
                    "• <b>Skewness:</b> Asymmetry (+ = right tail, - = left tail)<br>" +
                    "• <b>Kurtosis:</b> Peak sharpness and tail weight"
                ),
                showarrow=False,
                font=dict(size=12),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(0, 0, 0, 0.3)",
                borderwidth=1,
                borderpad=4
            ),
            dict(
                x=0.22, y=0.15,  # Adjusted for bottom right
                xref="paper", yref="paper",
                text=(
                    "<b>KL Divergence Matrix:</b><br>" +
                    "• Measures how one distribution differs from another.<br>" +
                    "• Darker colors indicate greater differences.<br>" +
                    "• Diagonal is zero (distribution compared to itself).<br>" +
                    "• Note: KL divergence is not symmetric - A→B ≠ B→A."
                ),
                showarrow=False,
                font=dict(size=12),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(0, 0, 0, 0.3)",
                borderwidth=1,
                borderpad=4
            )
        ]
        
        for annotation in descriptions:
            fig.add_annotation(annotation)
        
        # Update margins to accommodate annotations
        fig.update_layout(
            margin=dict(t=0, l=0, r=0, b=0),  # Adjusted margins for better spacing
        )
        
        return fig
    
    def _get_statistics_table_data(self):
        """Get statistics for the table display"""
        distributions = {
            'Gamma': self.gamma,
            'Exponential': self.exp,
            'Poisson': self.poisson
        }
        
        names = []
        means = []
        variances = []
        skewness = []
        kurtosis = []
        
        for name, sample in distributions.items():
            names.append(name)
            means.append(f"{np.mean(sample):.2f}")
            variances.append(f"{np.var(sample):.2f}")
            skewness.append(f"{self._calculate_skewness(sample):.2f}")
            kurtosis.append(f"{self._calculate_kurtosis(sample):.2f}")
        
        return [names, means, variances, skewness, kurtosis]
    
    def _get_kl_divergence_matrix(self):
        """Calculate KL divergence matrix for all distribution pairs"""
        distributions = {
            'Gamma': self.gamma,
            'Exponential': self.exp,
            'Poisson': self.poisson
        }
        labels = list(distributions.keys())
        n = len(labels)
        kl_matrix = np.zeros((n, n))
        
        for i, (name1, sample1) in enumerate(distributions.items()):
            for j, (name2, sample2) in enumerate(distributions.items()):
                kl_matrix[i, j] = self.calculate_kl_divergence(sample1, sample2)
        
        return kl_matrix, labels
    
    def _gamma_pdf(self, x, shape, scale):
        """Compute gamma PDF"""
        return (x**(shape-1) * np.exp(-x/scale) / 
                (scale**shape * math.gamma(shape)))
    
    def _exponential_pdf(self, x, scale):
        """Compute exponential PDF"""
        return (1/scale) * np.exp(-x/scale)
    
    def _poisson_pmf(self, k, lambda_):
        """Compute Poisson PMF"""
        return np.exp(-lambda_) * lambda_**k / np.array([math.factorial(int(ki)) for ki in k])
    
    def _calculate_skewness(self, x):
        """Calculate the skewness of a distribution"""
        return np.mean(((x - np.mean(x))/np.std(x))**3)
    
    def _calculate_kurtosis(self, x):
        """Calculate the kurtosis of a distribution"""
        return np.mean(((x - np.mean(x))/np.std(x))**4) - 3  # -3 for excess kurtosis

# Create and run the comparison
comparison = DistributionComparison(sample_size=1000)
fig = comparison.create_dashboard()
fig.show()
