import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from astropy.table import Table
from astropy.io import fits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Column
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

filename = '/beegfs/car/njm/OUTPUT/PRIMVS.fits'
# Load FITS file
data = Table.read(fits.open(filename, memmap=True)[1], format='fits')

# Load FITS file
spicydata = Table.read('/beegfs/car/njm/OUTPUT/PRIMVS_spicyclass1.fits', format='fits')
data_sourceids = np.array(data['sourceid'])
spicydata_sourceids = np.array(spicydata['sourceid'])
common_sourceids = np.intersect1d(data_sourceids, spicydata_sourceids)
subset_data_mask = np.isin(data_sourceids, common_sourceids)



# Define the manually assigned weights
# Define the list of feature names you want to use

selected_feature_names = ['l','b','parallax','pmra','pmdec','Z-K','Y-K','J-K','H-K',
			'mag_avg','Cody_M','Cody_Q','stet_k','eta','eta_e','med_BRP',
			'range_cum_sum','max_slope','MAD','mean_var','percent_amp',
			'true_amplitude','roms','p_to_p_var','lag_auto','AD',
			'std_nxs','weight_mean','weight_std','weight_skew','weight_kurt',
			'mean','std','skew','kurt','time_range','true_period']

feature_weights = [0.5,0.5,0.3,0.4,0.4,0.9,0.9,0.9,0.9,0.7,0.6,0.4,0.4,0.4,0.5,0.5,0.4,0.7,0.6,0.8,0.7,0.6,0.5,0.5,0.7,0.5,0.4,0.4,0.6,0.6,0.6,0.6,0.4,0.4,0.2,0.9]
#feature_weights = [0.5, 0.5, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.6, 0.9]





selected_features = data[selected_feature_names]

# Convert selected features to Pandas DataFrame
df = selected_features.to_pandas()

#Convert missing values to 0
df.fillna(0, inplace=True)

# Convert DataFrame to NumPy array
features = df.values

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Convert scaled features to PyTorch tensors
features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
weights_tensor = torch.tensor(feature_weights, dtype=torch.float32)

# Multiply the normalized features by the weights
data_features_tensor = features_tensor * weights_tensor


# Step 1: Prepare the training data
train_features = data_features_tensor[subset_data_mask]






# Step 2: Define and train the One-Class SVM model
model = OneClassSVM(gamma='auto')  # Adjust the 'nu' parameter as needed
model.fit(train_features)  # Train the model

# Step 3: Make predictions on the dataset
predictions = model.predict(features_tensor)  # Predict anomaly labels for dataset

# Step 6: Add the predicted class column to the original data
predicted_class_column = Column(name='SVM_predicted_class', data=predictions)
data.add_column(predicted_class_column)




# Step 2: Define and fit the Local Outlier Factor model
model = LocalOutlierFactor(n_neighbors=10, novelty=True)
model.fit(train_features)

# Step 3: Make predictions on the dataset
predictions = model.predict(data_features_tensor)

# Step 4: Add the predicted class column to the original data
predicted_class_column = Column(name='LOF_predicted_class', data=predictions)
data.add_column(predicted_class_column)




# Step 2: Define and train the Isolation Forest model
model = IsolationForest(contamination=0.1)  # Adjust the contamination parameter as needed
model.fit(train_features)  # Train the model

# Step 3: Make predictions on the dataset
predictions = model.predict(features_tensor)  # Predict anomaly labels for dataset

# Step 5: Add the predicted class column to the original data
predicted_class_column = Column(name='IF_predicted_class', data=predictions)
data.add_column(predicted_class_column)




# Step 2: Define and train the Gaussian Mixture Model
model = GaussianMixture(n_components=2)  # Adjust the number of components as needed
model.fit(train_features)  # Train the model

# Step 3: Calculate the log-likelihood of each instance
log_likelihoods = model.score_samples(features_tensor)  # Calculate log-likelihoods

# Step 6: Add the predicted class column to the original data
predicted_class_column = Column(name='GMM_predicted_class', data=log_likelihoods)
data.add_column(predicted_class_column)


# Write the data table to a CSV file
data.write('/beegfs/car/njm/PRIMVS/autoencoder/forest_class1.fits', format='fits', overwrite=True)


