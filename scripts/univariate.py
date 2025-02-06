import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from logger import logger  # Import logger

def summary_statistics(df, name="Dataset"):
    """Prints summary statistics and logs the step."""
    logger.info(f"Generating summary statistics for {name}")
    print(f"\nSummary Statistics for {name}:\n", df.describe())

def plot_histograms(df, name="Dataset"):
    """Plots histograms for numerical columns and logs the step."""
    logger.info(f"Plotting histograms for {name}")
    df.hist(figsize=(12, 8), bins=30)
    plt.suptitle(f"{name} - Feature Distributions")
    plt.show()

def plot_boxplots(df, numeric_columns, name="Dataset"):
    """Plots boxplots for numeric columns and logs the step."""
    for col in numeric_columns:
        logger.info(f"Plotting boxplot for {col} in {name}")
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col} - {name}")
        plt.show()
