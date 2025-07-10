import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('train_cleaned_titanic.csv')

# Principal Component Analysis (PCA) is a technique used to reduce the dimensionality of a dataset while preserving as much variance as possible.
# PCA finds the directions (principal components) that maximize the variance in the data.

# assume df is a cleaned DataFrame with only numerical values and no target column
X = df.drop('Survived', axis=1)

# standardize features before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# apply PCA
pca = PCA(n_components=None)  # keep all components initially
X_pca = pca.fit_transform(X_scaled)

# plot explained variance ratio
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         pca.explained_variance_ratio_.cumsum(), marker='o')
plt.xlabel('number of principal components')
plt.ylabel('cumulative explained variance')
plt.title('pca explained variance')
plt.grid(True)
plt.tight_layout()
plt.show()

# the explained variance ratio shows how much variance each principal component explains. high variance = more info
# to reduce dimensionality, you might choose the number of components that explain, say, 95% of the variance:
pca_95 = PCA(n_components=0.95)
X_reduced = pca_95.fit_transform(X_scaled)
print(f"reduced to {X_reduced.shape[1]} components")

#PCA doesn't know which features are relevant to prediction
# it just captures the variance in the data. so it's good for unsupervised learning tasks.
# (exploratory analysis, noise reduction, or visualization)