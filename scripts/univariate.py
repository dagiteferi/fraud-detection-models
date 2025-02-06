import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.logger import logger # Import logger

def summary_statistics(df, name="Dataset"):
    """Prints summary statistics and logs the step."""
    logger.info(f"Generating summary statistics for {name}")
    print(f"\nSummary Statistics for {name}:\n", df.describe())

def plot_histograms(df, name="Dataset"):
    """Plots histograms for numerical columns and logs the step."""
    logger.info(f"Plotting histograms for {name}")
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
    df.hist(ax=axes, bins=30)
    for ax in axes.flatten():
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    plt.suptitle(f"{name} - Feature Distributions")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_boxplots(df, numeric_columns, name="Dataset"):
    """Plots boxplots for numeric columns and logs the step."""
    for col in numeric_columns:
        logger.info(f"Plotting boxplot for {col} in {name}")
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col} - {name}")
        plt.show()
