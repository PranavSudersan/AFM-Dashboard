import numpy as np
from sklearn.cluster import KMeans
import itertools

def hypotenuse(a,b):
    return np.sqrt(np.square(a) + np.square(b))

#line-wise flattening of image along the x (fast) axis. order=0 subtracts each line by its mean (simple offset). 
#order=1,2...subtract a fitted polynomial of that order to each line
def flatten_line(data, order, pos_shift=True):
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
    data_flattened = data['Z']-offset
    if pos_shift == True:
        data_flattened = data_flattened - data_flattened.min() #shift to make all values positive
    return data_flattened


#generate Matrix to use with lstsq for levelling
def poly_matrix(x, y, order=2):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x**i * y**j
    return G

#flatten image by subtracting a fitted plane of "order" on "points" array
#TODO: include point selection window, add modes for automatic (kmeans) or manual (thresholding/points) point selection
def flatten_plane(img_data, points, order=1):
    # X,Y = np.meshgrid(self.df_matrix.columns,
    #                   self.df_matrix.index)
    X, Y = np.meshgrid(img_data['X'], img_data['Y'])

    if order == 1:
        # best-fit linear plane
        A = np.c_[points[:,0], points[:,1], np.ones(points.shape[0])]
        C,_,_,_ = np.linalg.lstsq(A, points[:,2], rcond=None)    # coefficients
        #print(C)
        # evaluate it on grid
        Z_zerofit = C[0]*X + C[1]*Y + C[2]
##            print(Z)
        # self.df['Zero fit'] = C[0]*self.df[self.plot_x] + \
        #                       C[1]*self.df[self.plot_y] + C[2]
        #print(self.df)
    elif order == 2:
        x, y, z = points.T
        #x, y = x - x[0], y - y[0]  # this improves accuracy

        # make Matrix:
        G = poly_matrix(x, y, order)
        # Solve for np.dot(G, m) = z:
        m = np.linalg.lstsq(G, z, rcond=None)[0]
        #print('m', m)
        # Evaluate it on a grid...
##            GG = self.poly_matrix(X.ravel(), Y.ravel(), order)
##            Z = np.reshape(np.dot(GG, m), X.shape)
##            print(Z)
        df['Zero fit'] = np.polynomial.polynomial.polyval2d(df[plot_x],
                                                                 df[plot_y],
                                                                 np.reshape(m, (-1, 3)))
    
    # df[plot_z+' corrected'] = df[plot_z]-df['Zero fit']
    Z_leveled = img_data['Z'] - Z_zerofit
    
    #organize data into matrix for heatmap plot
    # self.df_matrix = self.df.pivot_table(values=self.plot_z+' corrected',
    #                                      index=self.plot_y,
    #                                      columns=self.plot_x,
    #                                      aggfunc='first')
    return Z_leveled - Z_leveled.min()


#cluster image data using kmeans clustering algorithm. Returns a label image of the same shape, 
#where label numbers correspond to each cluster: 0, 1, 2 etc. The label values are sorted in same order as
#cluster center values in ascending order
def segment_kmeans(data, n_clusters):
    chan_data = data['Z']
    kmeans = KMeans(n_clusters=n_clusters) # Create a KMeans instance with 2 clusters: kmeans
    data_reshaped = chan_data.reshape(-1, 1)
    kmeans.fit(data_reshaped)
    labels = kmeans.labels_
    # Get cluster centers and sort their indices in ascending order
    sorted_cluster_indices = np.argsort(kmeans.cluster_centers_.flatten())    
    # Create a mapping from original labels to new labels
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_cluster_indices)}    
    # Apply mapping to relabel clusters
    sorted_labels = np.vectorize(label_mapping.get)(kmeans.labels_)
    return sorted_labels.reshape(chan_data.shape)
    # return labels.reshape(chan_data.shape)
