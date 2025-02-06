import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.logger import logger  # Import logger

def summary_statistics(df, name="Dataset"):
    """Prints summary statistics and logs the step."""
    logger.info(f"Generating summary statistics for {name}")
    print(f"\nSummary Statistics for {name}:\n", df.describe())

def plot_histograms(df, name="Dataset"):
    """Plots histograms for numerical columns and logs the step."""
    logger.info(f"Plotting histograms for {name}")

    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Create subplots
    num_plots = len(numerical_cols)
    num_rows = (num_plots + 2) // 3  # Calculate the number of rows needed, rounded up
    fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5 * num_rows))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, col in enumerate(numerical_cols):
        df[col].hist(ax=axes[i], bins=30)
        axes[i].set_title(f"Distribution of {col}")
        for label in axes[i].get_xticklabels():
            label.set_rotation(45)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

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
