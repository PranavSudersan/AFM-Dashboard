import numpy as np

def hypotenuse(a,b):
    return np.sqrt(np.square(a) + np.square(b))

#line-wise flattening of image along the x (fast) axis. order=0 subtracts each line by its mean (simple offset). 
#order=1,2...subtract a fitted polynomial of that order to each line
def flatten_line(data, order):
    if order == 0:
        offset = np.atleast_2d(np.mean(data['Z'], axis=1)).T
    else:
        p, res, rank, sing, rcond = np.polyfit(data['X'], data['Z'].T, 1, full=True)
        offset = np.array([np.poly1d(p[:,i])(data['X']) for i in range(p.shape[1])])
    return data['Z']-offset
