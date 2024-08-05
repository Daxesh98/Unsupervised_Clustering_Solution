import matplotlib.pyplot as plt
import seaborn as sns

def Pair_Plot(df):
    """Plot pairwise relationships in the dataset."""
    sns.pairplot(df[['Age', 'Annual_Income', 'Spending_Score']])
    plt.suptitle('Pair Plot of Features', y=1.02)  # Adjust title position
    plt.show()
    
def scatter_clusters(df):
    """Scatter plot of clusters."""
    sns.scatterplot(x='Annual_Income', y='Spending_Score', data=df, hue='Cluster', palette='colorblind')
    plt.title('Clusters Visualization')
    plt.show()

def plot_elbow(wss):
    """Plot the Elbow method to determine the optimal number of clusters."""
    plt.plot(wss['cluster'], wss['WSS_Score'], marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WSS Score')
    plt.title('Elbow Method for Optimal Clusters')
    plt.grid(True)  # Add grid for better readability
    plt.show()

def plot_silhouette(silhouette_scores):
    """Plot the Silhouette Score to evaluate cluster quality."""
    plt.plot(silhouette_scores['cluster'], silhouette_scores['Silhouette_Score'], marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal Clusters')
    plt.grid(True)  # Add grid for better readability
    plt.show()

def plot_silhouette_score_final(variables3):
    """Plot the silhouette score for the final set of features."""
    variables3.plot(x='cluster', y='Silhouette_Score', marker='o', linestyle='-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score with Final Features')
    plt.grid(True)  # Add grid for better readability
    plt.show()
