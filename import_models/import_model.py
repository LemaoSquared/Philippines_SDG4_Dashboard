import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib

class ModelHandler():

    def __init__(self):
        df = pd.read_csv('data_wrangling\\Cleaned_Philippines_Education_Statistics.csv')
        df_filtered = df[df['Cohort_Survival_Rate'] > 0]
        self.regional_profile = df_filtered.groupby('Geolocation').agg({
            'Participation_Rate': 'mean',
            'Completion_Rate': 'mean',
            'Cohort_Survival_Rate': 'mean',
        }).reset_index()

        self.cluster_scaler = joblib.load('import_models\\scaler.pkl') 
        self.loaded_kmeans = joblib.load('import_models\\kmeans_model.pkl')
        self.regional_profile['clusters'] = self.loaded_kmeans.fit_predict(
            self.cluster_scaler.transform(
                self.regional_profile[['Participation_Rate', 'Completion_Rate', 'Cohort_Survival_Rate']]
            )
        )
    
    def get_clusters(self):
        """ 
        A method that returns a dataframe containing the clusters for each region.

        """
        # 1. Load data
        return self.regional_profile
