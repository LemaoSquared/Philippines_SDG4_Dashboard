import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib

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

# 4. Create and Fit the Model
# We start with 3 clusters (High, Mid, and Low Performing regions)
wcss = []
ss = []
dbi = []
chi = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=77, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    wcss.append(kmeans.inertia_)
    if i > 1:
        ss.append(silhouette_score(X_scaled, labels))
        dbi.append(davies_bouldin_score(X_scaled, labels))
        chi.append(calinski_harabasz_score(X_scaled, labels))

# Comparing results

# WCSS: Distance of each point to each cluster 
#     (The smaller the decrease = the more closer each point is to the centroid)
print(wcss)
plt.clf()
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method (WCSS)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Silhouette Score: How good the clusters have been made
#     (1 = each point is far from other groups)
#     (Imagine it as the reverse WCSS, checks distance from other groups)
print(ss)
plt.clf()
plt.plot(range(2, 11), ss, marker='o', color='orange')
plt.title('Silhouette Score (1 is the best)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.show()

# Davies-Bouldin Index 
#     Looks at each point's compactness to the centroid and separation to the other clusters
#     Lower is better
print(dbi)
plt.clf()
plt.plot(range(2, 11), dbi, marker='o', color='red')
plt.title('Davies-Bouldin Index (Find the Minimum)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('DBI Score')
plt.show()

# Calinski Harabasz Score
# How close each point are inside each cluster and the distance from cluster to cluster
# The higher the better
print(chi)
plt.clf()
plt.plot(range(2, 11), chi, marker='o', color='blue')
plt.title('Calinski-Harabasz Index (Find the Peak)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('CH Score')
plt.show()

# 5. Interpret the Clusters

# You can also run this code dave if complex ang sa import model shinanneganz
print("--- Cluster Averages ---")
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

kmeans = KMeans(n_clusters=4, random_state=77, n_init=10, verbose=0)
kmeans.fit(X_scaled)
joblib.dump(kmeans, 'import_models//kmeans_model.pkl') # Saves the model
regional_profile['Cluster'] = kmeans.predict(X_scaled)
joblib.dump(scaler, 'import_models//scaler.pkl') # Saves the model
# print(regional_profile.groupby('Cluster')[features].mean())

# 6. Visualize the results
plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=regional_profile, 
    x='Participation_Rate', 
    y='Cohort_Survival_Rate', 
    hue='Cluster', 
    palette='Set1', 
    s=150
)

# Add Labels to the points
for i in range(regional_profile.shape[0]):
    plt.text(
        regional_profile.Participation_Rate[i], 
        regional_profile.Cohort_Survival_Rate[i], 
        regional_profile.Geolocation[i],
        fontsize=9
    )

plt.title('Regional Profiling: Participation vs. Survival Clusters')
plt.show()