import seaborn as sns
import matplotlib.pyplot as plt
from scripts.logger import logger # Import logger

def correlation_heatmap(df, name="Dataset"):
    """Plots correlation heatmap and logs the step."""
    logger.info(f"Generating correlation heatmap for {name}")
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title(f"Correlation Heatmap - {name}")
    plt.show()

def plot_boxplot(df, x_column, y_column, name="Dataset"):
    """Plots boxplot of one numerical feature grouped by a categorical feature."""
    logger.info(f"Generating boxplot for {y_column} by {x_column} in {name}")
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[x_column], y=df[y_column])
    plt.title(f"{y_column} by {x_column} - {name}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()

def pairplot_features(df, selected_features, hue_column, name="Dataset"):
    """Plots pairplots for selected features and logs the step."""
    logger.info(f"Generating pairplot for {name}")
    sns.pairplot(df[selected_features], hue=hue_column, diag_kind="kde")
    plt.suptitle(f"Pairplot - {name}")
    plt.show()
