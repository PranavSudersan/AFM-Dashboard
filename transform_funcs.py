import numpy as np
from sklearn.cluster import KMeans

def hypotenuse(a,b):
    return np.sqrt(np.square(a) + np.square(b))

#line-wise flattening of image along the x (fast) axis. order=0 subtracts each line by its mean (simple offset). 
#order=1,2...subtract a fitted polynomial of that order to each line
def flatten_line(data, order):
    if len(data['Z'].shape) == 1:
        if order == 0:
            offset = np.mean(data['Z'])
        else:
            p, res, rank, sing, rcond = np.polyfit(data['X'], data['Z'], order, full=True)
            offset = np.poly1d(p)(data['X'])
    else:
        if order == 0:
            offset = np.atleast_2d(np.mean(data['Z'], axis=1)).T
        else:
            p, res, rank, sing, rcond = np.polyfit(data['X'], data['Z'].T, order, full=True)
            offset = np.array([np.poly1d(p[:,i])(data['X']) for i in range(p.shape[1])])
    return data['Z']-offset

#cluster image data using kmeans clustering algorithm. Returns a label image of the same shape, 
#where label numbers correspond to each cluster: 0, 1, 2 etc.
def segment_kmeans(data, n_clusters):
    chan_data = data['Z']
    kmeans = KMeans(n_clusters=n_clusters) # Create a KMeans instance with 2 clusters: kmeans
    data_reshaped = chan_data.reshape(-1, 1)
    kmeans.fit(data_reshaped)
    labels = kmeans.labels_
    return labels.reshape(chan_data.shape)
