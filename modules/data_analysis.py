import seaborn as sns
import matplotlib.pyplot as plt

def analyze_data(data, config):
    # Plot correlation heatmap
    correlation_threshold = config["correlation_threshold"]
    corr = data.corr()
    sns.heatmap(corr, annot=True)
    plt.show()

    # Print strong correlations
    strong_corrs = (corr > correlation_threshold).sum()
    print(f"Strong correlations: {strong_corrs}")
