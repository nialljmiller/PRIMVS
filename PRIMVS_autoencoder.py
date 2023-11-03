import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from astropy.table import Table
from astropy.io import fits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

filename = '/beegfs/car/njm/OUTPUT/PRIMVS.fits'
outputfp = '/beegfs/car/njm/PRIMVS/autoencoder/'
# Load FITS file
data = Table.read(fits.open(filename, memmap=True)[1], format='fits')

# Define the list of feature names you want to use
selected_feature_names = ['l','b','parallax','pmra','pmdec','Z-K','Y-K','J-K','H-K','mag_avg','Cody_M','stet_k','eta','eta_e','med_BRP','range_cum_sum','max_slope','MAD','mean_var','percent_amp','true_amplitude','roms','p_to_p_var','lag_auto','AD','std_nxs','weight_mean','weight_std','weight_skew','weight_kurt','mean','std','skew','kurt','time_range','true_period']
feature_weights = [0.5,0.5,0.3,0.4,0.4,0.9,0.9,0.9,0.9,0.7,0.6,0.4,0.4,0.4,0.5,0.5,0.4,0.7,0.6,0.8,0.7,0.6,0.5,0.5,0.7,0.5,0.4,0.4,0.6,0.6,0.6,0.6,0.4,0.4,0.2,0.9]
selected_features = data[selected_feature_names]

# Convert selected features to Pandas DataFrame
df = selected_features.to_pandas()

# Convert missing values to 0
df.fillna(0, inplace=True)

# Convert DataFrame to NumPy array
features = df.values

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Convert scaled features to PyTorch tensors
features_tensor = torch.tensor(scaled_features, dtype=torch.float32)


# Convert the feature weights to a PyTorch tensor
weights_tensor = torch.tensor(feature_weights, dtype=torch.float32)

# Multiply the normalized features by the weights
features_tensor = features_tensor * weights_tensor




# Split the data into training and testing sets
X_train, X_test = train_test_split(features_tensor, test_size=0.2, random_state=42)

# Define the dimensions of the input data
input_dim = X_train.size(1)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),   
            nn.ReLU(),
            nn.Linear(128, 64),          
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),          
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Create the autoencoder model
autoencoder = Autoencoder(input_dim, latent_dim=32)

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Define a learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# Create DataLoader for training and testing sets
train_dataset = TensorDataset(X_train, X_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test, X_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



# Define your training loop
def train_model(num_epochs):
    # Initialize an empty list to store the training loss values
    train_losses = []
     
    for epoch in range(num_epochs):
        train_loss = 0.0

        for batch_features, _ in train_loader:
            optimizer.zero_grad()
            outputs = autoencoder(batch_features)
            loss = criterion(outputs, batch_features)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Calculate the average training loss for the current epoch
        avg_train_loss = train_loss / len(train_loader)

        # Append the average training loss to the list
        train_losses.append(avg_train_loss)

        # Update the learning rate based on the training loss
        scheduler.step(avg_train_loss)

        # Print the average training loss for each epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

    # Plot the loss values
    plt.plot(range(1, num_epochs+1), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs')
    plt.savefig('/beegfs/car/njm/OUTPUT/Loss.jpg', dpi=300, bbox_inches='tight')
    plt.clf()

# Load a saved model
def load_model(model_path, autoencoder):
    autoencoder.load_state_dict(torch.load(model_path))
    autoencoder.eval()
    return autoencoder


Train_model = False
if Train_model == True:
    # Train the model
    num_epochs = 1000
    train_model(num_epochs)
    # Save the model
    torch.save(autoencoder.state_dict(), 'autoencoder.pth')
else:
    # Example usage to load a saved model
    model_path = 'autoencoder.pth'
    autoencoder = load_model(model_path, autoencoder)


# Obtain the latent space representation
with torch.no_grad():
    encoded_data = autoencoder.encoder(features_tensor).numpy()




# Load FITS file
spicydata = Table.read('/beegfs/car/njm/OUTPUT/PRIMVS_spicyclass1.fits', format='fits')
spicysubclass_ids = np.array(spicydata['sourceid'])
spicy_subclass1_indices = [idx for idx, obj_id in enumerate(data['sourceid']) if obj_id in spicysubclass_ids]


# Load FITS file
spicydata = Table.read('/beegfs/car/njm/OUTPUT/PRIMVS_spicyclass2.fits', format='fits')
spicysubclass_ids = np.array(spicydata['sourceid'])
spicy_subclass2_indices = [idx for idx, obj_id in enumerate(data['sourceid']) if obj_id in spicysubclass_ids]


# Load FITS file
spicydata = Table.read('/beegfs/car/njm/OUTPUT/PRIMVS_spicyclass3.fits', format='fits')
spicysubclass_ids = np.array(spicydata['sourceid'])
spicy_subclass3_indices = [idx for idx, obj_id in enumerate(data['sourceid']) if obj_id in spicysubclass_ids]


# Load FITS file
newdata = Table.read('/beegfs/car/njm/PRIMVS/autoencoder/forest_class1.fits', format='fits')

# Calculate the threshold value for the top 10%
threshold = np.percentile(newdata['GMM_predicted_class'], 99)
mask = newdata['GMM_predicted_class'] >= threshold
newdata = newdata[mask]

mask = newdata['IF_predicted_class'] == 1
newdata = newdata[mask]

mask = newdata['LOF_predicted_class'] == 1
newdata = newdata[mask]

newsubclass_ids = np.array(newdata['sourceid'])
new_subclass1_indices = [idx for idx, obj_id in enumerate(data['sourceid']) if obj_id in newsubclass_ids]


# Load FITS file
newdata = Table.read('/beegfs/car/njm/PRIMVS/autoencoder/forest_class2.fits', format='fits')

# Calculate the threshold value for the top 10%
threshold = np.percentile(newdata['GMM_predicted_class'], 99)
mask = newdata['GMM_predicted_class'] >= threshold
newdata = newdata[mask]

mask = newdata['IF_predicted_class'] == 1
newdata = newdata[mask]

mask = newdata['LOF_predicted_class'] == 1
newdata = newdata[mask]

newsubclass_ids = np.array(newdata['sourceid'])
new_subclass2_indices = [idx for idx, obj_id in enumerate(data['sourceid']) if obj_id in newsubclass_ids]


# Load FITS file
newdata = Table.read('/beegfs/car/njm/PRIMVS/autoencoder/forest_class3.fits', format='fits')

# Calculate the threshold value for the top 10%
threshold = np.percentile(newdata['GMM_predicted_class'], 99)
mask = newdata['GMM_predicted_class'] >= threshold
newdata = newdata[mask]

mask = newdata['IF_predicted_class'] == 1
newdata = newdata[mask]

mask = newdata['LOF_predicted_class'] == 1
newdata = newdata[mask]

newsubclass_ids = np.array(newdata['sourceid'])
new_subclass3_indices = [idx for idx, obj_id in enumerate(data['sourceid']) if obj_id in newsubclass_ids]

print(len(new_subclass1_indices) + len(new_subclass2_indices) + len(new_subclass3_indices))


plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c='#103811', alpha=0.3, s=2)
plt.scatter(encoded_data[spicy_subclass1_indices, 0], encoded_data[spicy_subclass1_indices, 1], label='Class 1', c='#80d2af', alpha=0.1, s=5)
plt.scatter(encoded_data[new_subclass1_indices, 0], encoded_data[new_subclass1_indices, 1], label='New Class 1', c='#4e73d6', alpha=0.1, s=5)
plt.scatter(encoded_data[spicy_subclass2_indices, 0], encoded_data[spicy_subclass2_indices, 1], label='Class 2', c='#80d2af', alpha=0.1, s=5)
plt.scatter(encoded_data[new_subclass2_indices, 0], encoded_data[new_subclass2_indices, 1], label='New Class 2', c='#4e73d6', alpha=0.4, s=5)
plt.scatter(encoded_data[spicy_subclass3_indices, 0], encoded_data[spicy_subclass3_indices, 1], label='Class 3', c='#80d2af', alpha=0.4, s=5)
plt.scatter(encoded_data[new_subclass3_indices, 0], encoded_data[new_subclass3_indices, 1], label='New Class 3', c='#4e73d6', alpha=0.4, s=5)
plt.xlabel('Latent Dimension 0')
plt.ylabel('Latent Dimension 1')
plt.title('Latent Space Representation')
plt.legend()
plt.savefig(outputfp + 'LatentDim_0_1.jpg', dpi=300, bbox_inches='tight')
plt.clf()



# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)  # Set the number of components as desired
latent_pca = pca.fit_transform(encoded_data)

# Plot the latent space in 3D
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(latent_pca[:, 0], latent_pca[:, 1], c='k', alpha=0.2, s=2)
ax.scatter(latent_pca[spicy_subclass1_indices, 0], latent_pca[spicy_subclass1_indices, 1], label='Class 1', c='darkred', alpha=0.5, s=5)
ax.scatter(latent_pca[spicy_subclass2_indices, 0], latent_pca[spicy_subclass2_indices, 1], label='Class 2', c='darkgreen', alpha=0.5, s=5)
ax.scatter(latent_pca[spicy_subclass3_indices, 0], latent_pca[spicy_subclass3_indices, 1], label='Class 3', c='darkblue', alpha=0.5, s=5)
ax.scatter(latent_pca[new_subclass1_indices, 0], latent_pca[new_subclass1_indices, 1], label='New Class 1', c='r', alpha=0.4, s=6)
ax.scatter(latent_pca[new_subclass2_indices, 0], latent_pca[new_subclass2_indices, 1], label='New Class 2', c='g', alpha=0.4, s=6)
ax.scatter(latent_pca[new_subclass3_indices, 0], latent_pca[new_subclass3_indices, 1], label='New Class 3', c='b', alpha=0.4, s=6)
plt.legend()
ax.set_xlabel('Latent Dimension 1')
ax.set_ylabel('Latent Dimension 2')
ax.set_xlim(-2, 1)
ax.set_ylim(-2, 1)
plt.title('Latent Space Representation (PCA)')
plt.savefig(outputfp + 'LatentPCA_0_1.jpg', dpi=300, bbox_inches='tight')
plt.clf()





# Apply PCA for dimensionality reduction
pca = PCA(n_components=3)  # Set the number of components as desired
latent_pca = pca.fit_transform(encoded_data)

# Plot the latent space in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(latent_pca[:, 0], latent_pca[:, 1], latent_pca[:, 2], c='#103811', alpha=0.01, s=2)
ax.scatter(latent_pca[spicy_subclass1_indices, 0], latent_pca[spicy_subclass1_indices, 1], latent_pca[spicy_subclass1_indices, 2], label='Class 1', c='#80d2af', alpha=0.4, s=5)
ax.scatter(latent_pca[new_subclass1_indices, 0], latent_pca[new_subclass1_indices, 1], latent_pca[new_subclass1_indices, 2], label='New Class 1', c='#4e73d6', alpha=0.4, s=5)
ax.scatter(latent_pca[spicy_subclass2_indices, 0], latent_pca[spicy_subclass2_indices, 1], latent_pca[spicy_subclass2_indices, 2], label='Class 2', c='#80d2af', alpha=0.4, s=5)
ax.scatter(latent_pca[new_subclass2_indices, 0], latent_pca[new_subclass2_indices, 1], latent_pca[new_subclass2_indices, 2], label='New Class 2', c='#4e73d6', alpha=0.4, s=5)
ax.scatter(latent_pca[spicy_subclass3_indices, 0], latent_pca[spicy_subclass3_indices, 1], latent_pca[spicy_subclass3_indices, 2], label='Class 3', c='#80d2af', alpha=0.4, s=5)
ax.scatter(latent_pca[new_subclass3_indices, 0], latent_pca[new_subclass3_indices, 1], latent_pca[new_subclass3_indices, 2], label='New Class 3', c='#4e73d6', alpha=0.4, s=5)
plt.legend()
ax.set_xlabel('Latent Dimension 1')
ax.set_ylabel('Latent Dimension 2')
ax.set_zlabel('Latent Dimension 3')
ax.set_xlim(-2, 1)
ax.set_ylim(-2, 1)
ax.set_zlim(-2, 1)
plt.title('Latent Space Representation (PCA)')
plt.savefig(outputfp + 'LatentPCA_0_1_2.jpg', dpi=300, bbox_inches='tight')
plt.clf()


for alpha in [0.1,0.05,0.01]:
    for s in [1,2,5]:
        print(s,alpha)
        # Create a figure and subplot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        scatter = ax.scatter(latent_pca[:, 0], latent_pca[:, 1], latent_pca[:, 2], c='grey', alpha=alpha, s=s)
        scatter_class1 = ax.scatter(latent_pca[spicy_subclass1_indices, 0], latent_pca[spicy_subclass1_indices, 1], latent_pca[spicy_subclass1_indices, 2], label='Class 1', c='tomato', alpha=0.01, s=2)
        scatter_class2 = ax.scatter(latent_pca[spicy_subclass2_indices, 0], latent_pca[spicy_subclass2_indices, 1], latent_pca[spicy_subclass2_indices, 2], label='Class 2', c='lightgreen', alpha=0.01, s=2)
        scatter_class3 = ax.scatter(latent_pca[spicy_subclass3_indices, 0], latent_pca[spicy_subclass3_indices, 1], latent_pca[spicy_subclass3_indices, 2], label='Class 3', c='lightblue', alpha=0.01, s=2)
        scatter_newclass1 = ax.scatter(latent_pca[new_subclass1_indices, 0], latent_pca[new_subclass1_indices, 1], latent_pca[new_subclass1_indices, 2], label='New Class 1', c='g', alpha=0.7, s=5)
        scatter_newclass2 = ax.scatter(latent_pca[new_subclass2_indices, 0], latent_pca[new_subclass2_indices, 1], latent_pca[new_subclass2_indices, 2], label='New Class 2', c='g', alpha=0.7, s=5)
        scatter_newclass3 = ax.scatter(latent_pca[new_subclass3_indices, 0], latent_pca[new_subclass3_indices, 1], latent_pca[new_subclass3_indices, 2], label='New Class 3', c='b', alpha=0.7, s=5)

        # Function to update the plot for each frame
        def update(frame):
            ax.view_init(elev=20*(abs(frame-180)/180), azim=frame)  # Change the azimuth angle
            return scatter, scatter_class1, scatter_newclass1, scatter_class2, scatter_newclass2, scatter_class3, scatter_newclass3

        # Create the animation
        animation = FuncAnimation(fig, update, frames=range(0, 360, 2), interval=50)

        ax.set_xlim(-2, 4)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        # Save the animation as a GIF
        animation.save(outputfp + str(alpha*100) + str(s) + 'latent_space_rotation.gif', writer='imagemagick')

'''
# Create a figure and subplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
scatter = ax.scatter(latent_pca[:, 0], latent_pca[:, 1], latent_pca[:, 2], c='grey', alpha=0.001, s=1)
scatter_class1 = ax.scatter(latent_pca[spicy_subclass1_indices, 0], latent_pca[spicy_subclass1_indices, 1], latent_pca[spicy_subclass1_indices, 2], label='Class 1', c='tomato', alpha=0.1, s=2)
scatter_class2 = ax.scatter(latent_pca[spicy_subclass2_indices, 0], latent_pca[spicy_subclass2_indices, 1], latent_pca[spicy_subclass2_indices, 2], label='Class 2', c='lightgreen', alpha=0.1, s=2)
scatter_class3 = ax.scatter(latent_pca[spicy_subclass3_indices, 0], latent_pca[spicy_subclass3_indices, 1], latent_pca[spicy_subclass3_indices, 2], label='Class 3', c='lightblue', alpha=0.1, s=2)
scatter_newclass1 = ax.scatter(latent_pca[new_subclass1_indices, 0], latent_pca[new_subclass1_indices, 1], latent_pca[new_subclass1_indices, 2], label='New Class 1', c='g', alpha=0.7, s=5)
scatter_newclass2 = ax.scatter(latent_pca[new_subclass2_indices, 0], latent_pca[new_subclass2_indices, 1], latent_pca[new_subclass2_indices, 2], label='New Class 2', c='g', alpha=0.7, s=5)
scatter_newclass3 = ax.scatter(latent_pca[new_subclass3_indices, 0], latent_pca[new_subclass3_indices, 1], latent_pca[new_subclass3_indices, 2], label='New Class 3', c='b', alpha=0.7, s=5)

# Set plot limits and labels
#ax.set_xlabel('Latent Dimension 1')
#ax.set_ylabel('Latent Dimension 2')
#ax.set_zlabel('Latent Dimension 3')
ax.set_xlim(-2, 1)
ax.set_ylim(-1.5, 1)
ax.set_zlim(-1, 1)
#plt.title('Latent Space Representation (PCA)')

# Function to update the plot for each frame
def update(frame):
    ax.view_init(elev=20*(abs(frame-180)/180), azim=frame)  # Change the azimuth angle
    return scatter, scatter_class1, scatter_newclass1, scatter_class2, scatter_newclass2, scatter_class3, scatter_newclass3

# Create the animation
animation = FuncAnimation(fig, update, frames=range(0, 360, 2), interval=50)

# Save the animation as a GIF
animation.save(outputfp + 'latent_space_rotation_crop.gif', writer='imagemagick')
'''

