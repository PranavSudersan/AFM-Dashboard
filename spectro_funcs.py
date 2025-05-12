import numpy as np
import scipy.ndimage as ndimage
from scipy.optimize import curve_fit

import fit_funcs as ftf

#calculated adhesion from minima. 'index' output used to calculated sample deformation later. 'zero' used for zero value
def adhesion(force_data, segment, method, zero_pts, min_percentile, fit_order):
    # segment = 'retract'
    data_x, data_y = force_data[segment]['x'], force_data[segment]['y']
    ind_min = np.argmin(data_y)
    if ind_min != 0:
        ind_maxs = np.where(data_y[:ind_min]>=np.percentile(data_y[:ind_min],min_percentile))[0]
        if len(ind_maxs) <= fit_order+1:
            method = 'simple'
    else:
         method = 'simple'
    
    if method == 'simple':
        f_zero = np.median(force_data['approach']['y'][:zero_pts]) #CHECK THIS
        fadh_ymin = data_y[ind_min]
        adhesion = f_zero - fadh_ymin
        fit_x = np.array([data_x[0], data_x[ind_min], data_x[ind_min]])
        fit_y = np.array([f_zero, f_zero, fadh_ymin])
        if 'd' in force_data[segment].keys():
            data_d = force_data[segment]['d']
            fit_d = np.array([data_d[0], data_d[ind_min], data_d[ind_min]])
            data_z = force_data[segment]['z']
            fit_z = np.array([data_z[0], data_z[ind_min], data_z[ind_min]])
            return {'value': adhesion, 'segment': segment, 'x': fit_x, 'd': fit_d, 'z': fit_z,
                    'y': fit_y, 'zero': f_zero, 'min': fadh_ymin, 'index': ind_min}
        else:
            return {'value': adhesion, 'segment': segment, 'x': fit_x, 'y': fit_y, 
                    'zero': f_zero, 'min': fadh_ymin, 'index': ind_min}
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

        f_zero = np.median(data_y[:zero_pts]) #CHECK THIS
        # f_zero = zero_y.mean()
        # f_min = data_y[ind_min]
        if 'd' in force_data[segment].keys():
            data_d = force_data[segment]['d']
            zero_d = np.linspace(data_d[0], data_d[ind_min], 100)
            fadh_d = data_d[ind_min]
            fit_d = np.append(zero_d, [fadh_d, fadh_d])
            data_z = force_data[segment]['z']
            zero_z = np.linspace(data_z[0], data_z[ind_min], 100)
            fadh_z = data_z[ind_min]
            fit_z = np.append(zero_z, [fadh_z, fadh_z])
            return {'value': adhesion, 'segment': segment,  'x': fit_x, 'd': fit_d, 'z': fit_z,
                    'y': fit_y, 'zero': f_zero, 'min': fadh_ymin, 'index': ind_min}
        else:
            return {'value': adhesion, 'segment': segment,  'x': fit_x, 'y': fit_y, 
                    'zero': f_zero, 'min': fadh_ymin, 'index': ind_min}
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

def snapin(defl_data, segment, method, min_percentile, fit_order, back_pts, findmax, zero):
    from wsxm_analyze import set_funcdict_kwargs #set kwargs for other params from a parameter output
    # min_percentile = 1
    # fit_order = 2
    # segment = 'approach'
    data_x, data_y = defl_data[segment]['x'][1:], defl_data[segment]['y'][1:] #discard first point
    # test2_x, test2_y = test2['Z'], test2['ZZ'][:,y_ind,x_ind]
    # test_y_filt = ndimage.median_filter(test_y, size=filter_size) #filter
    # test_y_filt_sobel = ndimage.sobel(test_y_filt) #sobel transform
    # #this method works well when the jump in points is very fast, no points in between.
    # n_data = len(test_x)
    # tol_ind = int(thresh*n_data) #tolerance
    if method == 'minima':
        ind_min = np.argmin(data_y)
        # ind_min = np.argmax(test_y_filt_sobel)

        # amp_sobel = ndimage.sobel(test_y[:ind_min+tol_ind]) #sobel transform
        # # amp_sobel = ndimage.sobel(test_y) #sobel transform
        # # amp_sobel = ndimage.sobel(test_y_filt) #sobel transform
        # # ind_max = np.argmax(amp_sobel)
        # try:
        if ind_min == 0:
            return {'value': 0, 'segment': segment, 'x': [], 'y': [], 'd':[], 'z': [],
                    'index_min': 0, 'zero': 0}
        else:
            ind_maxs = np.where(data_y[:ind_min]>=np.percentile(data_y[:ind_min],min_percentile))[0]
            if len(ind_maxs) <= fit_order+1:
                return {'value': 0, 'segment': segment, 'x': [], 'y': [], 'd':[], 'z': [],
                        'index_min': 0, 'zero': 0}
            else:
                # # testmin_x, testmin_y = test_x[ind_max], test_y[ind_max]
                # # poly = np.poly1d([-amp_sobel[ind_max], testmin_y-(-amp_sobel[ind_max]*testmin_x)])
                # # poly = np.poly1d([slope_avg, testmin_y-(slope_avg*testmin_x)])
                # if len(ind_maxs) == 1:
                #     slope_avg = -amp_sobel[ind_maxs].mean()
                #     testmin_x, testmin_y = test_x[ind_maxs].mean(), test_y[ind_maxs].mean()
                #     poly = np.poly1d([slope_avg, testmin_y-(slope_avg*testmin_x)])
                # else:
                #fit zero line with polynomial
                p, res, rank, sing, rcond = np.polyfit(data_x[ind_maxs], data_y[ind_maxs], fit_order, full=True)
                poly = np.poly1d(p)

                fit_x_all = np.linspace(data_x[0], data_x[ind_min], 100)
                fit_y_all = poly(fit_x_all)

                snapin_x = data_x[ind_min]
                snapin_y0 = data_y[ind_min]
                snapin_y1 = poly(snapin_x) #zero deflection point at snapin
                snapin_distance = snapin_y1-snapin_y0

                fit_x = np.append(fit_x_all, [snapin_x, snapin_x])
                fit_y = np.append(fit_y_all, [snapin_y0, snapin_y1])
                set_funcdict_kwargs(channel='Normal force',param='Stiffness',kwargs={'snapin_index': ind_min})
                if 'd' in defl_data[segment].keys():
                    data_d = defl_data[segment]['d']
                    fit_d_all = np.linspace(data_d[0], data_d[ind_min], 100)
                    snapin_d = data_d[ind_min]
                    fit_d = np.append(fit_d_all, [snapin_d, snapin_d])
                    data_z = defl_data[segment]['z']
                    fit_z_all = np.linspace(data_z[0], data_z[ind_min], 100)
                    snapin_z = data_z[ind_min]
                    fit_z = np.append(fit_z_all, [snapin_z, snapin_z])
                    return {'value': snapin_distance, 'segment': segment, 'x': fit_x, 'd':fit_d, 'z': fit_z, 
                            'y': fit_y, 'index_min': ind_min, 'zero': snapin_y1} #TODO: ADD index_surf EVERYWHERE!
                else:
                    return {'value': snapin_distance, 'segment': segment, 'x': fit_x, 'y': fit_y, 
                            'index_min': ind_min, 'zero': snapin_y1}
    elif method == 'gradient':
        data_y_sobel = ndimage.sobel(data_y) #sobel transform
        ind_max = np.argmax(data_y_sobel)
        if ind_max == 0 or findmax == False: #set findmax=False to find global minima in gradient and ignore max gradient value
            ind_max = len(data_y)
        ind_min = np.argmin(data_y_sobel[:ind_max]) #point of snapin
        if ind_min == 0:
            return {'value': 0, 'segment': segment, 'x': [], 'y': [], 'd':[], 'z': [], 
                    'index_surf': 0, 'index_min': 0, 'zero': 0}
        else:
            if back_pts > ind_min:
                back_pts = ind_min
            if zero == 'max': #set method by which "zero deflection" point is estimated 
                zero_y =  data_y[ind_min:ind_min-back_pts:-1].max() #max of back_pts before index of high gradient
            elif zero == 'mean':
                zero_y =  np.mean(data_y[ind_min:ind_min-back_pts:-1]) #mean of back_pts before index of high gradient
            elif zero == 'median':
                zero_y =  np.median(data_y[ind_min:ind_min-back_pts:-1]) #median of back_pts before index of high gradient
            elif zero == 'ini':
                zero_y = np.median(data_y[:back_pts]) #median of inital "back_pts" number of points of data_y, ignores normal deflection drift here unlike others
            snap_indmin = np.argmin(data_y[ind_min:ind_max])+ind_min #point of tip-sample contact
            snapin_distance = (zero_y - data_y[snap_indmin]) + (data_x[ind_min-1] - data_x[snap_indmin])
            #surface found by finding x which is snapin_distance length to the left of point of snapin (ind_min-1)
            # surf_ind = ind_min-1 + np.argwhere(data_x[ind_min-1:]<(data_x[ind_min-1]-snapin_distance))[0][0] 
            surf_ind = np.argmin(abs(data_x-(data_x[ind_min-1]-snapin_distance)))
            # print((data_x[ind_min-1]-snapin_distance), snapin_distance, data_x[ind_min-1], data_x[surf_ind], data_y[snap_indmin], zero_y)
            fit_x = np.array([data_x[ind_min-1], data_x[ind_min-1], data_x[surf_ind]])
            fit_y = np.array([data_y[snap_indmin], zero_y, zero_y])
            set_funcdict_kwargs(channel='Normal force',param='Stiffness',kwargs={'snapin_index': snap_indmin})
            if 'd' in defl_data[segment].keys():
                data_d = defl_data[segment]['d']
                fit_d = np.array([data_d[ind_min-1], data_d[ind_min-1], data_d[snap_indmin]])
                data_z = defl_data[segment]['z']
                fit_z = np.array([data_z[ind_min-1], data_z[ind_min-1], data_z[surf_ind]])
                return {'value': snapin_distance, 'segment': segment, 'x': fit_x, 'd':fit_d, 'z': fit_z,
                        'y': fit_y, 'index_surf': surf_ind, 'index_min': snap_indmin, 'zero': zero_y}
            else:
                return {'value': snapin_distance, 'segment': segment, 'x': fit_x, 'y': fit_y, 
                        'index_surf': surf_ind, 'index_min': snap_indmin, 'zero': zero_y}
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

#for method: 'simple poly', fits a 2nd order polynomial on data after minima and returns the slope of tangent at the end point
# for method: 'best gradient', gradient of data is filered and fitting range for data is found based on percentile_range of the filtered gradient.
# the range is the longest chunk of consective data satifying the percentile condition. fit_order can be used to either do a linear fit or parabola fit
#over this range (tangent returned as slope for the case of parabola again)
def stiffness(force_data, segment, method, fit_order, snapin_index, filter_size, percentile_range):
    # segment = 'approach'
    # fit_order=2
    idx_min = snapin_index #np.argmin(force_data[segment]['y'])
    # try:
    if len(force_data[segment]['x'][idx_min:]) <= fit_order+1:
        return {'value': 0, 'segment': segment, 'x': [], 'y': [], 'd': [], 'x_surf': 0}
    else:          
        data_x, data_y = force_data[segment]['x'][idx_min:], force_data[segment]['y'][idx_min:]
        # print(len(force_data[segment]['x']), len(force_data[segment]['y']))
        # print(idx_min, len(data_x), len(data_y))
        if method == 'simple poly':
            p, res, rank, sing, rcond = np.polyfit(data_x, data_y, 2, full=True) #2nd order fit 
            poly1 = np.poly1d(p)
            x0, y0 = data_x[-1], poly1(data_x[-1])
            p_tan = [2*p[0]*x0+p[1], y0-(p[1]*x0)-(2*p[0]*x0**2)] #tangent slope equation
            poly2 = np.poly1d(p_tan)
            stiff_ind = slice(idx_min, len(data_x))
        elif method == 'best gradient':
            data_y_grad = np.gradient(data_y, data_x)
            data_y_gradfilt = ndimage.median_filter(data_y_grad, size=filter_size)#, mode='nearest')
            median_slope = np.median(data_y_gradfilt)
            data_y_good = (data_y_gradfilt<=np.percentile(data_y_gradfilt, percentile_range[1])) & (data_y_gradfilt>=np.percentile(data_y_gradfilt, percentile_range[0]))
            #find largest chunk of consecutive data following above condition
            ind_good = np.argwhere(data_y_good==True).flatten()
            ind_chunk = np.argwhere((np.diff(ind_good)-1).astype(bool)==True).flatten()
            ind_chunk_all = np.insert(ind_chunk+1, 0, 0)
            ind_chunk_all = np.append(ind_chunk_all, len(ind_good))
            argmax = np.argmax(np.diff(ind_chunk_all))
            arg_range = ind_chunk_all[argmax], ind_chunk_all[argmax+1]-1
            ind_fitrange = slice(ind_good[arg_range[0]], ind_good[arg_range[-1]]+1)
            stiff_ind = slice(idx_min+ind_good[arg_range[0]], idx_min+ind_good[arg_range[-1]]+1)
            # p, res, rank, sing, rcond = np.polyfit(data_x, data_y, fit_order, full=True) #2nd order fit 
            # print(p)
            if fit_order == 1:
                p, res, rank, sing, rcond = np.polyfit(data_x[ind_fitrange], data_y[ind_fitrange], 1, full=True) #linear fit
                p_tan = p
                poly2 = np.poly1d(p)
            elif fit_order == 2: #fit parabola with positive curvature
                poly2d = lambda x, a, b, c: (a*x**2) + (b*x) + c
                p, _ = curve_fit(poly2d, data_x[ind_fitrange], data_y[ind_fitrange], bounds=([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]))    
                poly1 = np.poly1d(p)
                x0, y0 = data_x[-1], poly1(data_x[-1])
                p_tan = [2*p[0]*x0+p[1], y0-(p[1]*x0)-(2*p[0]*x0**2)] #tangent slope equation
                poly2 = np.poly1d(p_tan)            
        n_data = len(data_x)
        fit_x_all = np.linspace(data_x[0], data_x[-1], n_data*10) #CHECK!!
        fit_y_all = poly2(fit_x_all)
        fitind_min = np.argmin(abs(fit_y_all-data_y.min()))
        fitind_max = np.argmin(abs(fit_y_all-data_y.max()))
        fit_x = fit_x_all[fitind_min:fitind_max]
        fit_y = fit_y_all[fitind_min:fitind_max]
        
        zero_y = np.median(force_data[segment]['y'][:10]) #zero force CHECK does not consider drift
        surf_x =  (poly2-zero_y).r[0] #surface x value, found by intersection of stiffness fit with zero force
        # print(fit_x, fit_y)
        if 'd' in force_data[segment].keys():
            data_d = force_data[segment]['d'][idx_min:]
            fit_d_all = np.linspace(data_d[0], data_d[-1], n_data*10)
            fit_d = fit_d_all[fitind_min:fitind_max]
            data_z = force_data[segment]['z'][idx_min:]
            fit_z_all = np.linspace(data_z[0], data_z[-1], n_data*10)
            fit_z = fit_z_all[fitind_min:fitind_max]
            return {'value': -p_tan[0], 'segment': segment, 'x': fit_x, 'd': fit_d, 'z': fit_z, 'y': fit_y, 'fit_index': stiff_ind, 'x_surf': surf_x}
        else:
            return {'value': -p_tan[0], 'segment': segment, 'x': fit_x, 'y': fit_y, 'fit_index': stiff_ind, 'x_surf': surf_x}
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
#method 'minmax' find the minima and maxima and locates the amplitude change in data as factor of it initial standard deviation.
#change_factor is that factor of change beyond std_dev and num_pts is the number of initial points used to estimate amp extrema
def ampslope(amp_data, segment, filter_size, method, max_percentile, change, num_pts, change_factor):
    # segment = 'approach'
    if segment not in amp_data.keys():
        return {'value': 0, 'segment': '', 'x': [], 'd': [], 'z': [], 'y': []}
    amp_data_x, amp_data_y = amp_data[segment]['x'][1:], amp_data[segment]['y'][1:]
    n_data = len(amp_data_x)
    amp_data_y_filt = ndimage.median_filter(amp_data_y, size=filter_size) #filter
    if method == 'minmax':
        # amp_min, amp_max = amp_data_y.min(), amp_data_y.max()
        amp_extrema = [np.mean(amp_data_y_filt[:num_pts]), np.mean(amp_data_y_filt[-num_pts:])]
        amp_min, amp_max = min(amp_extrema), max(amp_extrema)
        if amp_min == amp_extrema[0]:
            ind_min = 0
            ind_max = n_data
        else:
            ind_min = n_data
            ind_max = 0
        amp_dev = np.std(amp_data_y[:num_pts])
        amp_change = amp_max-amp_min
        ind_mid = np.argmin(abs(amp_data_y_filt-((amp_max+amp_min)/2)))
        amp_min_target = amp_min+(amp_dev*change_factor)
        amp_max_target = amp_max-(amp_dev*change_factor)
        if ind_mid == 0: #avoid ind_mid to be last points of data
            ind_mid = 1
        if ind_mid == n_data-1:
            ind_mid = n_data-2
        # print(ind_mid, ind_min, ind_max)
        if ind_mid > ind_min:
            ind1 = ind_mid - np.argwhere(amp_data_y[ind_mid::-1]<amp_min_target)[0][0] 
            ind2 = ind_mid + np.argwhere(amp_data_y[ind_mid:]>amp_max_target)[0][0] 
        else:
            ind1 = ind_mid - np.argwhere(amp_data_y[ind_mid::-1]>amp_max_target)[0][0] 
            ind2 = ind_mid + np.argwhere(amp_data_y[ind_mid:]<amp_min_target)[0][0]
        ind_list = [ind1, ind2]
        ind_list.sort()
        if ind_list[1]-ind_list[0]>=1:
            p, res, rank, sing, rcond = np.polyfit(amp_data_x[ind_list[0]:ind_list[1]+1], amp_data_y[ind_list[0]:ind_list[1]+1], 1, full=True)
            slope = p[0]
            poly = np.poly1d(p)
        else:
            return {'value': 0, 'segment': segment, 'x': [], 'd': [], 'z': [], 'y': []}
        # print(amp_data_y[ind1], amp_data_y[ind2], amp_min+(amp_dev*change_factor))
        # plt.plot(amp_data_x, amp_data_y,'r.')
        # plt.plot(amp_data_x[ind_list[0]:ind_list[1]], amp_data_y[ind_list[0]:ind_list[1]],'b')
    else:
        amp_data_y_filt_sobel = ndimage.sobel(amp_data_y_filt) #sobel transform on filtered data

        #this method works well when the jump in points is very fast, no points in between.
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
    if 'd' in amp_data[segment].keys(): #IMPROVE!
        amp_data_d = amp_data[segment]['d']
        fitind3 = np.argmin(abs(amp_data_d-fit_x_all[fitind_min]))
        fitind4 = np.argmin(abs(amp_data_d-fit_x_all[fitind_max]))
        # fit_d = np.linspace(amp_data_d[fitind3], amp_data_d[fitind4], len(fit_y))
        # fit_d_all = np.interp(np.linspace(0,1,n_data*10), np.linspace(0,1,len(amp_data_d)), amp_data_d)
        fit_d_all = np.linspace(amp_data_d[0], amp_data_d[-1], n_data*10) #DOES NOT WORK WELL WITH 'd'
        fit_d = fit_d_all[fitind_min:fitind_max]
        amp_data_z = amp_data[segment]['z']
        fitind3 = np.argmin(abs(amp_data_z-fit_x_all[fitind_min]))
        fitind4 = np.argmin(abs(amp_data_z-fit_x_all[fitind_max]))
        # fit_d = np.linspace(amp_data_d[fitind3], amp_data_d[fitind4], len(fit_y))
        # fit_d_all = np.interp(np.linspace(0,1,n_data*10), np.linspace(0,1,len(amp_data_d)), amp_data_d)
        fit_z_all = np.linspace(amp_data_z[0], amp_data_z[-1], n_data*10) #DOES NOT WORK WELL WITH 'd'
        fit_z = fit_z_all[fitind_min:fitind_max]
        return {'value': abs(slope), 'segment': segment, 'x': fit_x, 'd': fit_d, 'z': fit_z, 'y': fit_y}
    else:
        return {'value': abs(slope), 'segment': segment, 'x': fit_x, 'y': fit_y}

#sigmoidal fit amplitude data to get "growth rate"
def ampgrowth(amp_data, segment, change):
    # segment = 'approach'
    if segment not in amp_data.keys():
        return {'value': 0, 'segment': '', 'x': [], 'd': [], 'z': [], 'y': []}
    amp_data_x, amp_data_y = amp_data[segment]['x'][1:], amp_data[segment]['y'][1:]
    amp_min, amp_max = amp_data_y.min(), amp_data_y.max()
    center_ind = np.argmin(abs(amp_data_y-((amp_min+amp_max)/2)))
    try:
        if change == 'up':
            
            p0 = [amp_max-amp_min,-amp_data_x[center_ind],1,amp_min] #initial guess
            popt, pcov = curve_fit(ftf.sigmoid, -amp_data_x, amp_data_y, p0, method='dogbox')    
            fit_x, fit_y = amp_data_x, ftf.sigmoid(-amp_data_x, *popt)
        elif change == 'down':
            p0 = [amp_max-amp_min,amp_data_x[center_ind],1,amp_min] #initial guess
            popt, pcov = curve_fit(ftf.sigmoid, amp_data_x, amp_data_y, p0, method='dogbox')    
            fit_x, fit_y = amp_data_x, ftf.sigmoid(amp_data_x, *popt)
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
        if 'd' in amp_data[segment].keys():
            fit_d = -amp_data[segment]['d']
            fit_z = -amp_data[segment]['z']
            return {'value': popt[2], 'segment': segment, 'x': fit_x, 'z': fit_z, 'd': fit_d, 'y': fit_y}
        else:
            return {'value': popt[2], 'segment': segment, 'x': fit_x, 'y': fit_y}
    except Exception as e:
        return {'value': 0, 'segment': segment, 'x': [], 'y': [], 'd': [], 'z': []}