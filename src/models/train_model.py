from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

def kmeans_model(df, x_col, y_col=None, n_clusters=5):
   
    if isinstance(x_col, list):
        features = df[x_col]  # Use multiple columns
    else:
        features = df[[x_col]]  # Use a single column
        
    if y_col is not None:
        features = features.join(df[[y_col]])  # Add y_col if provided

    kmodel = KMeans(n_clusters=n_clusters, random_state=42).fit(features)
    
    # Add cluster labels to the DataFrame
    df['Cluster'] = kmodel.labels_
    
    print(f"Cluster labels: {kmodel.labels_}")
    print("Cluster counts:")
    print(df['Cluster'].value_counts())
    
    return df, kmodel

def evaluate_kmeans(df, x_cols, k_range=range(3, 9)):
       
    wcss = []
    silhouette_scores = []
    
    for k in k_range:
        kmodel = KMeans(n_clusters=k, random_state=42).fit(df[x_cols])
        wcss.append({'cluster': k, 'WSS_Score': kmodel.inertia_})
        silhouette_scores.append({'cluster': k, 'Silhouette_Score': silhouette_score(df[x_cols], kmodel.labels_)})
    
    # Convert lists to DataFrames
    wss_df = pd.DataFrame(wcss)
    silhouette_df = pd.DataFrame(silhouette_scores)
    
    return wss_df, silhouette_df
