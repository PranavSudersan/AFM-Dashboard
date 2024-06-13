import numpy as np
import scipy.ndimage as ndimage
from scipy.optimize import curve_fit

import fit_funcs as ftf


def adhesion(force_data, method, zero_pts, min_percentile, fit_order):
    segment = 'retract'
    data_x, data_y = force_data[segment]['x'], force_data[segment]['y']
    ind_min = np.argmin(data_y)
    if ind_min != 0:
        ind_maxs = np.where(data_y[:ind_min]>=np.percentile(data_y[:ind_min],min_percentile))[0]
        if len(ind_maxs) <= fit_order+1:
            method = 'simple'
    else:
         method = 'simple'
    
    if method == 'simple':
        f_zero = data_y[:zero_pts].mean() #CHECK THIS
        fadh_ymin = data_y[ind_min]
        adhesion = f_zero - fadh_ymin
        fit_x = np.array([data_x[0], data_x[ind_min], data_x[ind_min]])
        fit_y = np.array([f_zero, f_zero, fadh_ymin])
        return {'value': adhesion, 'segment': segment, 'x': fit_x, 'y': fit_y, 'zero': f_zero, 'min': fadh_ymin}
    elif method == 'fitzero':
        # try:
        p, res, rank, sing, rcond = np.polyfit(data_x[ind_maxs], data_y[ind_maxs], fit_order, full=True)
        poly = np.poly1d(p)
        zero_x = np.linspace(data_x[0], data_x[ind_min], 100)
        zero_y = poly(zero_x)
        fadh_x = data_x[ind_min]
        fadh_ymin = data_y[ind_min]
        fadh_y0 = poly(fadh_x)
        adhesion = fadh_ymin-fadh_y0

        fit_x = np.append(zero_x, [fadh_x, fadh_x])
        fit_y = np.append(zero_y, [fadh_y0, fadh_ymin])

        f_zero = data_y[:zero_pts].mean() #CHECK THIS
        # f_zero = zero_y.mean()
        # f_min = data_y[ind_min]
        return {'value': adhesion, 'segment': segment,  'x': fit_x, 'y': fit_y, 'zero': f_zero, 'min': fadh_ymin}
        # except Exception as e:
        #     return {'value': 0, 'segment': segment, 'x': [], 'y': []}

# def func_snapin(defl_data, zero_pts): #CHECK ALGORITHM!
#     segment = 'approach'
#     # defl_sobel = ndimage.sobel(defl_data[segment]['y']) #sobel transform
#     # idx_min = np.argmin(defl_sobel) #id of sharpest corner in defl data
#     defl_idx_min = np.argmin(defl_data[segment]['y']) #id of minima of data
#     # defl_snapin = defl_data[segment]['y'][idx_min]
#     z_snapin = defl_data[segment]['x'][defl_idx_min]
#     defl_min = defl_data[segment]['y'][defl_idx_min]
#     defl_zero = defl_data[segment]['y'][:zero_pts].mean()
#     # z_min = defl_data['approach']['x'][defl_idx_min]
#     # print(idx_min)
#     return {'value': defl_zero - defl_min, 'segment': segment, 'x': [z_snapin, z_snapin], 'y': [defl_zero, defl_min]}

def snapin(defl_data, min_percentile, fit_order):
    # min_percentile = 1
    # fit_order = 2
    segment = 'approach'
    data_x, data_y = defl_data[segment]['x'], defl_data[segment]['y']
    # test2_x, test2_y = test2['Z'], test2['ZZ'][:,y_ind,x_ind]
    # test_y_filt = ndimage.median_filter(test_y, size=filter_size) #filter
    # test_y_filt_sobel = ndimage.sobel(test_y_filt) #sobel transform
    # #this method works well when the jump in points is very fast, no points in between.
    # n_data = len(test_x)
    # tol_ind = int(thresh*n_data) #tolerance
    ind_min = np.argmin(data_y)
    # ind_min = np.argmax(test_y_filt_sobel)

    # amp_sobel = ndimage.sobel(test_y[:ind_min+tol_ind]) #sobel transform
    # # amp_sobel = ndimage.sobel(test_y) #sobel transform
    # # amp_sobel = ndimage.sobel(test_y_filt) #sobel transform
    # # ind_max = np.argmax(amp_sobel)
    # try:
    if ind_min == 0:
        return {'value': 0, 'segment': segment, 'x': [], 'y': []}
    else:
        ind_maxs = np.where(data_y[:ind_min]>=np.percentile(data_y[:ind_min],min_percentile))[0]
        if len(ind_maxs) <= fit_order+1:
            return {'value': 0, 'segment': segment, 'x': [], 'y': []}
        else:
            # # testmin_x, testmin_y = test_x[ind_max], test_y[ind_max]
            # # poly = np.poly1d([-amp_sobel[ind_max], testmin_y-(-amp_sobel[ind_max]*testmin_x)])
            # # poly = np.poly1d([slope_avg, testmin_y-(slope_avg*testmin_x)])
            # if len(ind_maxs) == 1:
            #     slope_avg = -amp_sobel[ind_maxs].mean()
            #     testmin_x, testmin_y = test_x[ind_maxs].mean(), test_y[ind_maxs].mean()
            #     poly = np.poly1d([slope_avg, testmin_y-(slope_avg*testmin_x)])
            # else:
            p, res, rank, sing, rcond = np.polyfit(data_x[ind_maxs], data_y[ind_maxs], fit_order, full=True)
            poly = np.poly1d(p)

            fit_x_all = np.linspace(data_x[0], data_x[ind_min], 100)
            fit_y_all = poly(fit_x_all)

            snapin_x = data_x[ind_min]
            snapin_y0 = data_y[ind_min]
            snapin_y1 = poly(snapin_x)
            snapin_distance = snapin_y1-snapin_y0

            fit_x = np.append(fit_x_all, [snapin_x, snapin_x])
            fit_y = np.append(fit_y_all, [snapin_y0, snapin_y1])

            return {'value': snapin_distance, 'segment': segment, 'x': fit_x, 'y': fit_y, 'index': ind_min}
    # except Exception as e:
    #     return {'value': 0, 'segment': segment, 'x': [], 'y': []}

# def func_stiffness(force_data, bad_pts):
#     segment = 'approach'
#     idx_min = np.argmin(force_data[segment]['y'])
#     if idx_min == force_data[segment]['x'].shape[0]-1: #when spectra not good
#         return {'value': 0, 'segment': segment, 'x': force_data[segment]['x'][idx_min:], 'y': force_data[segment]['y'][idx_min:]}
#     else:
#         p, res, rank, sing, rcond = np.polyfit(force_data[segment]['x'][idx_min:], 
#                                                force_data[segment]['y'][idx_min:], 1, full=True)
#         poly = np.poly1d(p)
#         fit_data = {'x': force_data[segment]['x'][idx_min:], 'y': poly(force_data[segment]['x'][idx_min:])}
#         return {'value': -p[0], 'segment': segment, 'x': fit_data['x'], 'y': fit_data['y']}

#fits a 2nd order polynomial on data after minima and returns the slope of tangent at the end point
def stiffness(force_data, fit_order):
    segment = 'approach'
    fit_order=2
    idx_min = np.argmin(force_data[segment]['y'])
    # try:
    if len(force_data[segment]['x'][idx_min:]) <= fit_order+1:
        return {'value': 0, 'segment': segment, 'x': [], 'y': []}
    else:          
        data_x, data_y = force_data[segment]['x'][idx_min:], force_data[segment]['y'][idx_min:]
        p, res, rank, sing, rcond = np.polyfit(data_x, data_y, fit_order, full=True) #2nd order fit 
        poly1 = np.poly1d(p)
        x0, y0 = data_x[-1], poly1(data_x[-1])
        p_tan = [2*p[0]*x0+p[1], y0-(p[1]*x0)-(2*p[0]*x0**2)] #tangent slope equation
        poly2 = np.poly1d(p_tan)
        n_data = len(data_x)
        fit_x_all = np.linspace(data_x[0], data_y[-1], n_data*10)
        fit_y_all = poly2(fit_x_all)
        fitind_min = np.argmin(abs(fit_y_all-data_y.min()))
        fitind_max = np.argmin(abs(fit_y_all-data_y.max()))
        fit_x = fit_x_all[fitind_min:fitind_max]
        fit_y = fit_y_all[fitind_min:fitind_max]
        return {'value': -p_tan[0], 'segment': segment, 'x': fit_x, 'y': fit_y}
    # except Exception as e:
    #     return {'value': 0, 'segment': segment, 'x': [], 'y': []}

#slope of amplitude change during fd spectroscopy
# def func_ampslope(amp_data, range_factor):
#     segment = 'approach'
#     amp_sobel = ndimage.sobel(amp_data[segment]['y']) #sobel transform
#     # ind_max = np.argmax(amp_sobel)
#     # sobel_min, sobel_max = amp_sobel.min(), amp_sobel.max()
#     # #find range around sobel max to get fit range
#     # mid_sobel = sobel_min+(sobel_max-sobel_min)*range_factor
#     # ind_mid1 = np.argmin(abs(amp_sobel-mid_sobel))
#     # ind_diff = abs(ind_max-ind_mid1)
#     # ind_mid2 = ind_max + ind_diff if ind_mid1<ind_max else ind_max - ind_diff 
#     # ind_range = [min([ind_mid1, ind_mid2]), max([ind_mid1, ind_mid2])] #indices ordered from small to big

#     kmeans = KMeans(n_clusters=2) # Create a KMeans instance with 2 clusters: kmeans
#     kmeans.fit(amp_sobel.reshape(-1, 1)) 
#     centroids = kmeans.cluster_centers_
#     low_cluster, high_cluster = (0, 1) if centroids[0] < centroids[1] else (1, 0) #Get higher value clusters
#     labels = kmeans.labels_
#     high_cluster_data = amp_sobel[labels == high_cluster]
#     high_cluster_indices = np.where(labels.reshape(amp_sobel.shape) == high_cluster)[0]

#     # if ind_range[0] < 0 or ind_range[1] > len(amp_data[segment]['x'])-1:
#     if len(high_cluster_indices) <= 2: #ignore small clusters
#         return {'value': 0, 'segment': segment, 'x': np.array([]), 'y': np.array([])}
#     else:
#         p, res, rank, sing, rcond = np.polyfit(amp_data[segment]['x'][high_cluster_indices],
#                                                amp_data[segment]['y'][high_cluster_indices], 1, full=True)
#         poly = np.poly1d(p)
#         fit_data = {'x': amp_data[segment]['x'][high_cluster_indices], 'y': poly(amp_data[segment]['x'][high_cluster_indices])}
#         return {'value': -p[0], 'segment': segment, 'x': fit_data['x'], 'y': fit_data['y']}

#slope of amplitude change during fd spectroscopy
#amplitude data is filtered (using filter_size) and the high slope value above "max_percentile" are found by sobel transformation 
#method "average" returns the mean value of the slopes found, while method "fit" makes a linear fit on the original amplitude data
#where high slope values are found and returns the slope of the fit
def ampslope(amp_data, filter_size, method, max_percentile, change):
    segment = 'approach'  
    amp_data_x, amp_data_y = amp_data[segment]['x'], amp_data[segment]['y']
    amp_data_y_filt = ndimage.median_filter(amp_data_y, size=filter_size) #filter
    amp_data_y_filt_sobel = ndimage.sobel(amp_data_y_filt) #sobel transform on filtered data

    #this method works well when the jump in points is very fast, no points in between.
    n_data = len(amp_data_x)
    tol_ind = int(filter_size/4) #int(thresh*n_data) #tolerance
    if change == 'up': #for positive step changes (True Amplitude channel)
        ind_max = np.argmax(amp_data_y_filt_sobel)
        amp_sobel = ndimage.sobel(amp_data_y[:ind_max+tol_ind]) #sobel transform on actual data
        ind_maxs = np.where(amp_sobel>=np.percentile(amp_sobel,max_percentile))[0]
        ind_maxs = np.arange(ind_maxs.min(), ind_maxs.max()+1,1)
        if method == 'average' or len(ind_maxs)==1: #average to find slope
            slope = amp_sobel[ind_maxs].mean()
            ampmax_x, ampmax_y = amp_data_x[ind_maxs].mean(), amp_data_y[ind_maxs].mean()
            poly = np.poly1d([-slope, ampmax_y-(-slope*ampmax_x)])
        elif method == 'fit': #linear fit to find slope
            p, res, rank, sing, rcond = np.polyfit(amp_data_x[ind_maxs], amp_data_y[ind_maxs], 1, full=True)
            slope = p[0]
            poly = np.poly1d(p)
    elif change == 'down': #for negative step changes (Amplitude channel)
        ind_max = np.argmin(amp_data_y_filt_sobel)
        amp_sobel = ndimage.sobel(amp_data_y[:ind_max+tol_ind]) #sobel transform on actual data
        ind_maxs = np.where(amp_sobel<=np.percentile(amp_sobel,100-max_percentile))[0]
        ind_maxs = np.arange(ind_maxs.min(), ind_maxs.max()+1,1)
        if method == 'average' or len(ind_maxs)==1: #average to find slope
            slope = amp_sobel[ind_maxs].mean()
            ampmax_x, ampmax_y = amp_data_x[ind_maxs].mean(), amp_data_y[ind_maxs].mean()
            poly = np.poly1d([-slope, ampmax_y-(-slope*ampmax_x)])
        elif method == 'fit': #linear fit to find slope
            p, res, rank, sing, rcond = np.polyfit(amp_data_x[ind_maxs], amp_data_y[ind_maxs], 1, full=True)
            slope = p[0]
            poly = np.poly1d(p)

    fit_x_all = np.linspace(amp_data_x[0], amp_data_x[-1], n_data*10)
    fit_y_all = poly(fit_x_all)
    fitind1 = np.argmin(abs(fit_y_all-amp_data_y.min()))
    fitind2 = np.argmin(abs(fit_y_all-amp_data_y.max()))
    fitind_min = min([fitind1, fitind2])
    fitind_max = max([fitind1, fitind2])
    fit_x = fit_x_all[fitind_min:fitind_max]
    fit_y = fit_y_all[fitind_min:fitind_max]
    
    return {'value': abs(slope), 'segment': segment, 'x': fit_x, 'y': fit_y}

#sigmoidal fit amplitude data to get "growth rate"
def ampgrowth(amp_data):
    segment = 'approach'  
    amp_data_x, amp_data_y = amp_data[segment]['x'], amp_data[segment]['y']
    p0 = [max(amp_data_y), np.median(-amp_data_x),1,min(amp_data_y)] #initial guess
    try:
        popt, pcov = curve_fit(ftf.sigmoid, -amp_data_x, amp_data_y, p0, method='dogbox')    
        fit_x, fit_y = amp_data_x, ftf.sigmoid(-amp_data_x, *popt)
        #sigmoidal function method only works for spectroscopy with lots of points, not for force volume with less points
        #the maximum derivative method does not work for spectroscopy with lots of points due to "noise" during the points of max change
        # slope_max = popt[0]*popt[2]/4 #analytical slope maximum expression for sigmoidal curve (at x0)
        # print(popt, slope_max)
        # poly2 = np.poly1d([-slope_max, ((popt[0]/2)+popt[3])-(slope_max*popt[1])])
        # fit_x_all = np.linspace(test_x[0], test_x[-1], n_data*10)
        # fit_y_all = poly2(fit_x_all)
        # fitind_min = np.argmin(abs(fit_y_all-test_y.min()))
        # fitind_max = np.argmin(abs(fit_y_all-test_y.max()))
        # fit_x = fit_x_all[fitind_min:fitind_max]
        # fit_y = fit_y_all[fitind_min:fitind_max]
        return {'value': popt[2], 'segment': segment, 'x': fit_x, 'y': fit_y}
    except Exception as e:
        return {'value': 0, 'segment': segment, 'x': [], 'y': []}