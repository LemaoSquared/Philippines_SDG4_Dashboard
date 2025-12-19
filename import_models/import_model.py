import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib

class ModelHandler():

    def __init__(self):
        pass
    
    def get_clusters(self):
        # 1. Load data
        df = pd.read_csv('data_wrangling\\Cleaned_Philippines_Education_Statistics.csv')

        # 2. Preprocess: Group by Region to get a "Profile"
        # We filter out 0 values (like old Senior High data) to get accurate averages
        df_filtered = df[df['Cohort_Survival_Rate'] > 0]

        regional_profile = df_filtered.groupby('Geolocation').agg({
            'Participation_Rate': 'mean',
            'Completion_Rate': 'mean',
            'Cohort_Survival_Rate': 'mean',
        }).reset_index()

        # 3. Select Features & Scale
        # Scaling is CRITICAL because GPI is around 1.0 while Survival is around 80.0
        features = ['Participation_Rate', 'Completion_Rate', 'Cohort_Survival_Rate']
        X = regional_profile[features]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        loaded_kmeans = joblib.load('import_models\\kmeans_model.pkl')
        regional_profile['clusters'] = loaded_kmeans.fit_predict(X_scaled)
        return regional_profile
