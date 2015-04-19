from sklearn.decomposition import PCA
from dataset import load_or_create_dataset
import matplotlib.pyplot as plt

X, y = load_or_create_dataset(scaled=True, categorical=False)

# instantiate the model
model = PCA(n_components=2)

# fit the model: notice we don't pass the labels!
model.fit(X)

# transform the data to two dimensions
X_PCA = model.transform(X)
print ("shape of result:", X_PCA.shape)

# plot the results along with the labels
fig, ax = plt.subplots()
im = ax.scatter(X_PCA[:, 0],
                X_PCA[:, 1],
                # X_PCA[:, 2],
                c=y)
fig.colorbar(im)

plt.show()