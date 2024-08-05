import warnings
from src.data.make_dataset import load_data
from src.visualization.visualize import Pair_Plot, plot_elbow, plot_silhouette, scatter_clusters
from src.models.train_model import kmeans_model, evaluate_kmeans

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Load Data
    data_path = "data\\raw\\mall_customers.csv"
    df = load_data(data_path)
    
    # Data Analysis and Visualization
    # Plot a pairplot to understand feature distributions
    Pair_Plot(df)
        
    # Fit KMeans with initial parameters
    df, kmodel = kmeans_model(df, ['Annual_Income', 'Spending_Score'], y_col=None, n_clusters=5)
    
    # Add cluster labels to dataframe
    df['Cluster'] = kmodel.labels_
    
    # Visualize clusters
    scatter_clusters(df)
    
    # Elbow Method
    # Calculate WCSS for different number of clusters
    k_range = range(3, 9)
    wss, _ = evaluate_kmeans(df, ['Annual_Income', 'Spending_Score'], k_range)
    plot_elbow(wss)
    
    # Silhouette Measure
    _, silhouette_scores = evaluate_kmeans(df, ['Annual_Income', 'Spending_Score'], k_range)
    plot_silhouette(silhouette_scores)

    # Optimal Clustering with More Features
    # Train a model with additional features
    x_cols = ['Age', 'Annual_Income', 'Spending_Score']
    df, kmodel_final = kmeans_model(df, x_cols, y_col='Cluster', n_clusters=5)
    
    # Print the final clusters
    print(df.head())
