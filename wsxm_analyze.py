import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import signal
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import copy

def func_adhesion(force_data, method, zero_pts, min_percentile, fit_order):
    segment = 'retract'
    data_x, data_y = force_data[segment]['x'], force_data[segment]['y']
    ind_min = np.argmin(data_y)
    if method == 'simple':
        f_zero = data_y[-zero_pts:].mean() #CHECK THIS
        fadh_ymin = data_y[ind_min]
        adhesion = f_zero - fadh_ymin
        fit_x = np.array([data_x[-1], data_x[ind_min], data_x[ind_min]])
        fit_y = np.array([f_zero, f_zero, fadh_ymin])
        return {'value': adhesion, 'segment': segment, 'x': fit_x, 'y': fit_y, 'zero': f_zero, 'min': fadh_ymin}
    elif method == 'fitzero':
        try:
            ind_maxs = np.where(data_y[:ind_min]>=np.percentile(data_y[:ind_min],min_percentile))[0]
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
        except Exception as e:
            return {'value': 0, 'segment': segment, 'x': [], 'y': []}

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

def func_snapin(defl_data, min_percentile, fit_order):
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
    try:
        ind_maxs = np.where(data_y[:ind_min]>=np.percentile(data_y[:ind_min],min_percentile))[0]
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

        return {'value': snapin_distance, 'segment': segment, 'x': fit_x, 'y': fit_y}
    except Exception as e:
        return {'value': 0, 'segment': segment, 'x': [], 'y': []}

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
def func_stiffness(force_data):
    segment = 'approach'
    idx_min = np.argmin(force_data[segment]['y'])
    try:
        data_x, data_y = force_data[segment]['x'][idx_min:], force_data[segment]['y'][idx_min:]
        p, res, rank, sing, rcond = np.polyfit(data_x, data_y, 2, full=True) #2nd order fit 
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
    except Exception as e:
        return {'value': 0, 'segment': segment, 'x': [], 'y': []}

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
def func_ampslope(amp_data, filter_size, method, max_percentile):
    segment = 'approach'  
    amp_data_x, amp_data_y = amp_data[segment]['x'], amp_data[segment]['y']
    amp_data_y_filt = ndimage.median_filter(amp_data_y, size=filter_size) #filter
    amp_data_y_filt_sobel = ndimage.sobel(amp_data_y_filt) #sobel transform on filtered data
    #this method works well when the jump in points is very fast, no points in between.
    n_data = len(amp_data_x)
    tol_ind = int(filter_size/4) #int(thresh*n_data) #tolerance
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

    fit_x_all = np.linspace(amp_data_x[0], amp_data_x[-1], n_data*10)
    fit_y_all = poly(fit_x_all)
    fitind_min = np.argmin(abs(fit_y_all-amp_data_y.min()))
    fitind_max = np.argmin(abs(fit_y_all-amp_data_y.max()))
    fit_x = fit_x_all[fitind_min:fitind_max]
    fit_y = fit_y_all[fitind_min:fitind_max]
    
    return {'value': slope, 'segment': segment, 'x': fit_x, 'y': fit_y}

#sigmoidal fit amplitude data to get "growth rate"
def func_ampgrowth(amp_data):
    segment = 'approach'  
    amp_data_x, amp_data_y = amp_data[segment]['x'], amp_data[segment]['y']
    p0 = [max(amp_data_y), np.median(-amp_data_x),1,min(amp_data_y)] #initial guess
    try:
        popt, pcov = curve_fit(sigmoid, -amp_data_x, amp_data_y, p0, method='dogbox')    
        fit_x, fit_y = amp_data_x, sigmoid(-amp_data_x, *popt)
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

#TODO: PUT ALL FITTING FUNCTIONS (EG LORENTZIAN) IN A SEPARATE FILE
#general logistic function
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

#TODO: calibration dictionary to get in nm or nN from volts

global FUNC_DICT, CALIB_DICT, SPECT_DICT #CHECK THIS! TODO!

#dictionary of functions defined to extract spectroscopy data properties
#if function outputs other than 'value' is 'x', 'y', set plot type to 'line' below, else, set plot type to
#however it needs to be plotted as a dictionary for each additional output.
#TODO unit calibation add here also where necessary
FUNC_DICT = {'Normal force': {'Adhesion': {'function':func_adhesion,
                                           'kwargs': {'method': 'simple',
                                                      'zero_pts': 10,
                                                      'min_percentile': 1,
                                                      'fit_order': 2
                                                     },
                                           'plot type': 'line'#{'zero':'hline', 'min':'hline'}
                                           },
                              'Stiffness': {'function':func_stiffness,
                                            'kwargs': {#'bad_pts':1
                                                      },
                                            'plot type': 'line'
                                            },
                              'Snap-in distance': {'function':func_snapin,
                                                   'kwargs': {#'zero_pts': 10,
                                                              'min_percentile': 1, 
                                                              'fit_order': 2
                                                             },
                                                   'plot type': 'line'
                                                   }
                              },
             'Amplitude': {'Slope-amp':{'function':func_ampslope,
                                        'kwargs': {#'range_factor': 0.6,
                                                   'filter_size': 20,
                                                   'method': 'fit', #'fit','average'
                                                   'max_percentile': 99
                                                  },
                                        'plot type': 'line'
                                        },
                           'Growth rate':{'function':func_ampgrowth,
                                        'kwargs': {},
                                        'plot type': 'line'
                                        }
                          },
             'Excitation frequency': {},
             'Phase': {}
            }

# calibration dictionary for each channel. ADD MORE CHANNELS!
CALIB_DICT = {'Normal force': {'V': {'factor':1, 'offset':0}, 
                               'nm': {'factor':1, 'offset':0},
                               'nN':{'factor':1, 'offset':0}
                              },
              'Amplitude': {'V': {'factor':1, 'offset':0},
                            'nm': {'factor':1, 'offset':0},
                           },
              'Excitation frequency': {'V': {'factor':1, 'offset':0},
                                       'Hz': {'factor':1, 'offset':0}
                                       },
              'Phase': {'V': {'factor':1, 'offset':0}
                        }
             }

#rename spectroscopy line to standard names: approach and retract
SPECT_DICT = {'Forward':'approach', 'Backward': 'retract'} 

#update kwargs for FUNCT_DICT
def set_funcdict_kwargs(channel,param,kwargs):
    for key, value in kwargs.items():
        FUNC_DICT[channel][param]['kwargs'][key] = value

# Get spectroscopy data dictionary from force volume data at x,y
def wsxm_getspectro(data, channel, img_dir, x=0, y=0):
    # label_dict = {'Forward':'approach', 'Backward': 'retract'} #rename lines
    #initialize spectroscopy data dictionary
    # data_fd_dict = {'x': np.empty(0), 'y': np.empty(0), 'segment': np.empty(0)}
    #spectroscopy keys for the same image direction
    img_keys = [key for key in data[channel].keys() if key.startswith(f'Image {img_dir}')] 
    spectro_data = {}
    for key in img_keys:
        spectro_dir = SPECT_DICT[key.split(' ')[3]]
        x_pt = np.argmin(abs(data[channel][key]['data']['X']-x))
        y_pt = np.argmin(abs(data[channel][key]['data']['Y']-y))
        # if segment != 'both' and segment != spectro_dir: #skip unwanted segment
        #     continue
        # line_pts = int(data[channel][key]['header']['Number of points per ramp'])
        # data_fd_dict['x'] = np.append(data_fd_dict['x'], data[channel][key]['data']['Z'])
        # data_fd_dict['y'] = np.append(data_fd_dict['y'], data[channel][key]['data']['ZZ'][:,y,x])
        # data_fd_dict['segment'] = np.append(data_fd_dict['segment'], line_pts*[spectro_dir])
        spectro_data[spectro_dir] = {'y': data[channel][key]['data']['ZZ'][:,y_pt,x_pt],
                                     'x': data[channel][key]['data']['Z']}
    # data_fd = pd.DataFrame.from_dict(data_fd_dict)
    # #perform calculations for parameters (e.g. adhesion, stiffness, check FUNC_DICT) on the single spectroscopy curve
    # data_dict_param = {}
    # for param in FUNC_DICT[channel].keys():
    #     kwargs = FUNC_DICT[channel][param]['kwargs']
    #     _, data_dict_param[param] = FUNC_DICT[channel][param]['function'](spectro_data, **kwargs)   
        
    # print(x,y,data['ZZ'])
    return spectro_data

# Convert spectroscopy data dictionary to dataframe for plotting and calculate parameter
def wsxm_calcspectroparam(spectro_data, channel, unit, calc_params=True, properties=[]):
    #perform calculations for parameters (e.g. adhesion, stiffness, check FUNC_DICT) on the single spectroscopy curve
    # spectro_data = data[channel]['curves'][curv_num]['data']
    spectro_data_cali = copy.deepcopy(spectro_data)
    for key in spectro_data_cali.keys(): #calibrate
        spectro_data_cali[key]['y'] = (CALIB_DICT[channel][unit]['factor']*spectro_data_cali[key]['y']) + CALIB_DICT[channel][unit]['offset'] 
    df_spec = convert_spectro2df(spectro_data_cali) #pd.DataFrame.from_dict(data_fd_dict) #for plotting
    # print(channel, unit, CALIB_DICT[channel][unit])
    # df_spec['y'] = (CALIB_DICT[channel][unit]['factor']*df_spec['y']) + CALIB_DICT[channel][unit]['offset'] #calibrate
    data_dict_param = {}
    if calc_params==True:
        for param in FUNC_DICT[channel].keys():
            if param in properties:
                # if channel == FUNC_DICT[param]['channel']:
                kwargs = FUNC_DICT[channel][param]['kwargs']
                data_dict_param[param] = FUNC_DICT[channel][param]['function'](spectro_data_cali, **kwargs)
        
    return df_spec, data_dict_param

#convert spectro data dictionary to dataframe for plotting
def convert_spectro2df(data_dict):
    data = {'x': np.empty(0), 'y': np.empty(0), 'segment': np.empty(0)}
    for key, val in data_dict.items():
        for k in val.keys():
            data[k] = np.append(data[k], val[k])
            data_len = len(val[k])
        data['segment'] = np.append(data['segment'], [key]*data_len)        
    df_spec = pd.DataFrame.from_dict(data)
    return df_spec
    
#obtain property image from spectroscopy data of force-volume based on functions defined in FUNC_DICT
def calc_spectro_prop(data, properties=[]):
    # # img_keys = [key for key in data[channel].keys() if key.startswith(f'Image {img_dir}')] 
    # data_dict_spectro = {}
    # params = FUNC_DICT[channel].keys()
    # for param in params:
    #     data_dict_spectro[param] = np.empty(0)
    # #get common data from first channel e.g. number of points
    # # channel = FUNC_DICT[params[0]]['channel']
    # key = list(data[channel].keys())[0]
    # x_pts = int(data[channel][key]['header']['Number of rows'])
    # y_pts = int(data[channel][key]['header']['Number of columns'])
    # for y in range(y_pts):
    #     for x in range(x_pts):
    #         for param in params:
    #             # channel = FUNC_DICT[channel][param]['channel']
    #             kwargs = FUNC_DICT[channel][param]['kwargs']
    #             img_keys = [key for key in data[channel].keys() if key.startswith(f'Image {img_dir}')]
    #             spectro_data = {}
    #             for key in img_keys:
    #                 spectro_dir = SPECT_DICT[key.split(' ')[3]]
    #                 spectro_data[spectro_dir] = {'y': data[channel][key]['data']['ZZ'][:,y,x],
    #                                              'x': data[channel][key]['data']['Z']}
    #             param_result,_ = FUNC_DICT[channel][param]['function'](spectro_data, **kwargs)
    #             data_dict_spectro[param] = np.append(data_dict_spectro[param], [param_result])
    # # data_dict_spectro[spectro_dir] = adh_data.reshape(x_pts,y_pts)
    # for param in params:
    #     data_dict_spectro[param] = data_dict_spectro[param].reshape(x_pts, y_pts)

    # img_keys = [key for key in data[channel].keys() if key.startswith(f'Image {img_dir}')] 
    data_dict_param = {}
    chans = list(data.keys())
    chans.remove('Topography')
    for channel in chans:
        params = FUNC_DICT[channel].keys()
        for param in params:
            if param in properties:
                data_dict_param[param] = {}
                for chan_dir in data[channel].keys():
                    img_dir = chan_dir.split(' ')[1] #'Forward' or 'Backward'
                    if img_dir not in data_dict_param[param].keys():
                        data_dict_param[param][img_dir] = {'data': {'X': data[channel][chan_dir]['data']['X'], 
                                                                    'Y':  data[channel][chan_dir]['data']['Y'], 
                                                                    'Z': np.empty(0)},
                                                           'header':{}
                                                          }

                #get common data from first channel e.g. number of points
                # channel = FUNC_DICT[params[0]]['channel']
                # key = list(data[channel].keys())[0]
                        x_pts = int(data[channel][chan_dir]['header']['Number of rows [General Info]'])
                        y_pts = int(data[channel][chan_dir]['header']['Number of columns [General Info]'])
                        for y in range(y_pts):
                            for x in range(x_pts):
                                # for param in params:
                                    # channel = FUNC_DICT[channel][param]['channel']
                                kwargs = FUNC_DICT[channel][param]['kwargs']
                                img_keys = [key for key in data[channel].keys() if key.startswith(f'Image {img_dir}')]
                                spectro_data = {}
                                for key in img_keys:
                                    spectro_dir = SPECT_DICT[key.split(' ')[3]]
                                    spectro_data[spectro_dir] = {'y': data[channel][key]['data']['ZZ'][:,y,x],
                                                                 'x': data[channel][key]['data']['Z']}
                                param_result = FUNC_DICT[channel][param]['function'](spectro_data, **kwargs)
                                data_dict_param[param][img_dir]['data']['Z'] = np.append(data_dict_param[param][img_dir]['data']['Z'], 
                                                                                         [param_result['value']])
                print(f'{param} calculated')
                # data_dict_spectro[spectro_dir] = adh_data.reshape(x_pts,y_pts)
    for param in data_dict_param.keys():
        for img_dir in data_dict_param[param].keys():
            data_dict_param[param][img_dir]['data']['Z'] = data_dict_param[param][img_dir]['data']['Z'].reshape(x_pts, y_pts)    
    return data_dict_param

#get image data in appropriate matrix structure for plotting
def get_imgdata(data_dict_chan, style = 'XY', x=0, y=0, z=0):
    data0 = data_dict_chan['data'][style[0]]
    data1 = data_dict_chan['data'][style[1]]
    data_mat = np.meshgrid(data0, data1)
    if 'ZZ' in data_dict_chan['data'].keys(): #for force volume data
        if style == 'XY':
            z_pt = np.argmin(abs(data_dict_chan['data']['Z']-z))
            img_data = data_dict_chan['data']['ZZ'][z_pt,:,:] #1st index:xy sections, 2nd index:xz sections, 3rd index: yz sections
        elif style == 'XZ':
            y_pt = np.argmin(abs(data_dict_chan['data']['Y']-y))
            img_data = data_dict_chan['data']['ZZ'][:,y_pt,:]
        elif style == 'YZ':
            x_pt = np.argmin(abs(data_dict_chan['data']['X']-x))
            img_data = data_dict_chan['data']['ZZ'][:,:,x_pt]
    else: #for usual image data
        img_data = data_dict_chan['data']['Z']
    data_mat.append(img_data)
    return data_mat[0], data_mat[1], data_mat[2]

#get data at a specific line of an image. x=vertical line, y=horizontal line
def get_imgline(data_dict_chan, x=None, y=None):
    if x != None:
        x_pt = np.argmin(abs(data_dict_chan['data']['X']-x))
        return data_dict_chan['data']['Y'], data_dict_chan['data']['Z'][:,x_pt]
    if y != None:
        y_pt = np.argmin(abs(data_dict_chan['data']['Y']-y))
        return data_dict_chan['data']['X'], data_dict_chan['data']['Z'][y_pt,:]
        

        
def combine_forcevol_data(data, channel_list):
    output_all_dict = {}
    for img_dir in ['Forward', 'Backward']:
        output_data = {}
        z_data_temp = data[channel_list[0]][f'Image {img_dir} with Forward Ramps']['data']['Z']

        # z_data_full = np.concatenate([z_data, z_data]) #for both approach and retract for all channels
        x_len = len(data[channel_list[0]][f'Image {img_dir} with Forward Ramps']['data']['X'])
        y_len = len(data[channel_list[0]][f'Image {img_dir} with Forward Ramps']['data']['Y'])
        z_len = len(z_data_temp)
        # output_data['Z'] = np.reshape([[z_data_full]*(x_len*y_len)], (x_len,y_len,len(z_data_full))).flatten()
        specdir_list = []
        z_list = []
        z_array_dict = {'Forward': z_data_temp, 'Backward': np.flip(z_data_temp)}
        for spec_dir in ['Forward', 'Backward']:
            specdir_list.append([SPECT_DICT[spec_dir]]*x_len*y_len*z_len)  
            z_list.append(np.concatenate([z_array_dict[spec_dir]]*x_len*y_len))
        # print(z_data, z_list[0][:100])
        output_data['segment'] = np.concatenate(specdir_list)  
        output_data['Z'] = np.concatenate(z_list)  

        for chan in channel_list:
            output_data[chan] = []
            for spec_dir in ['Forward', 'Backward']:
                output_data[chan].append(data[chan][f'Image {img_dir} with {spec_dir} Ramps']['data']['ZZ'].flatten(order='F'))
                # print(len(output_data[chan]), output_data[chan][-1].shape,  len(output_data['segment']), len(output_data['segment'][-1]))
            output_data[chan] = np.concatenate(output_data[chan])

        output_df = pd.DataFrame(output_data)
        output_all_dict[img_dir] = output_df
    
#     output_df_a = output_df[output_df['segment']=='approach']
#     output_df_r = output_df[output_df['segment']=='retract']
#     plt.style.use("dark_background")
#     #adjust alpha of colormaps
#     cmap = plt.cm.Spectral
#     my_cmap = cmap(np.arange(cmap.N))
#     alphas = np.linspace(0.5,1, cmap.N)
#     # alphas = np.logspace(np.log10(0.5),0, cmap.N)
#     # alphas = np.ones(cmap.N)
#     alphas[0] = 0.2
#     alphas[1] = 0.3
#     alphas[2] = 0.4
#     BG = np.asarray([0.,0.,0.])
#     for i in range(cmap.N):
#         my_cmap[i,:-1] = my_cmap[i,:-1]*alphas[i]+BG*(1.-alphas[i])
#     my_cmap = ListedColormap(my_cmap)
#     plt.close('all')
#     fig1, ax1 = plt.subplots(6,2, figsize = (10, 30))
#     fig2, ax2 = plt.subplots(3,2, figsize = (10, 15))
#     k = 0
#     for i, col_i in enumerate(output_df.columns.drop('segment')):          
#         for j, col_j in enumerate(output_df.columns.drop('segment')):
#             if j > i:
#                 print(col_i, col_j)
#                 if col_i == 'Z':
#                     g = sns.lineplot(data=output_df_a, x=col_i, y=col_j, ax=ax2[k][0], estimator='median', errorbar=("pi",100))
#                     g = sns.lineplot(data=output_df_r, x=col_i, y=col_j, ax=ax2[k][1], estimator='median', errorbar=("pi",100))
#                     ax2[k][1].set_ylabel('')
#                 g = sns.histplot(data=output_df_a, x=col_i, y=col_j, bins=int(1.0*len(z_data_temp)), ax=ax1[k][0],
#                                  stat='frequency', edgecolor='none', linewidth=0, cmap=my_cmap)
#                 g = sns.histplot(data=output_df_r, x=col_i, y=col_j, bins=int(1.0*len(z_data_temp)), ax=ax1[k][1],
#                                  stat='frequency', edgecolor='none', linewidth=0, cmap=my_cmap)
#                 # g = sns.lineplot(data=output_df_a, x=col_i, y=col_j, ax=ax[k][0])
#                 # g = sns.lineplot(data=output_df_r, x=col_i, y=col_j, ax=ax[k][1])
#                 ax1[k][1].set_ylabel('')
#                 k += 1

#     ax1[0][0].set_title('approach')
#     ax1[0][1].set_title('retract')
#     ax2[0][0].set_title('approach')
#     ax2[0][1].set_title('retract')
#     fig_html1 = fig2html(fig1, plot_type='matplotlib', width=900, height=3000, pad=0.1)
#     fig_html2 = fig2html(fig2, plot_type='matplotlib', width=900, height=1500, pad=0.1)
    
#     plt.close()
#     # plt.show()
#     print('exit')
    return output_all_dict

#calibration functions

def get_psd_calib(data_dict):
    # im_data, head_data = read_wsxm_chan(filepath)
    # plt.close()
    # im_data = data_dict['data']
    head_data = data_dict['header']
    # print(head_data)
    #plot AFM Z image
    # xx = im_data['X'].reshape(128,128)
    # yy = im_data['Y'].reshape(128,128)
    # zz = im_data['Z']#.reshape(128,128)
    xx, yy, zz = get_imgdata(data_dict)
    plt.pcolormesh(xx,yy,zz, cmap='afmhot')    
    plt.colorbar()
    fig = fig2html(plt.gcf())
    plt.close()
    # plt.show()
    
    #Obtain Power Spectral Density of data
    #sample_rate = 2*num_pts*float(head_data['X-Frequency'].split(' ')[0])
    sample_rate = float(head_data['Sampling frequency [Miscellaneous]'].split(' ')[0])
    freq_array, z_pow = signal.periodogram(zz, sample_rate, scaling='density') #power spectral density
    z_pow_avg = np.average(z_pow, axis=0) #averaged
    z_pow_max = z_pow_avg.max()
    freq_drive = float(head_data['Resonance frequency [Dynamic settings]'].split(' ')[0])
    freq_array_shifted = freq_array + freq_drive
    # plt.plot(freq_array, z_pow_avg)
    # plt.show()

    z = zz.flatten()
    z_rms = np.sqrt(z.dot(z)/z.size)
    # print(zz.min(), zz.max(), z_rms)
    return freq_array_shifted, z_pow_avg, z_pow_max, z_rms, fig


#Lorentzian fit
# y0 = white noise offset, f0 = resonance freq, w = Full width at half maximum, A = area
def lorentzian(f, y0,f0, w, A):
    return y0 + ((2*A/np.pi) * (w / ( w**2 + 4*( f - f0 )**2)))


def get_calib(df_on, df_off, ind):
    freq_final = df_on['frequency'].iloc[ind]
    psd_final = df_on['psd'].iloc[ind] - df_off['psd'].iloc[ind]
    plt.plot(freq_final, psd_final)
    plt.plot(freq_final, df_on['psd'].iloc[ind])
    plt.plot(freq_final, df_off['psd'].iloc[ind])
    #plt.show()
    
    #guess = [0, 76000, 2000, 100000]
    y_guess = 0 #psd_final.min()
    f_guess = freq_final[psd_final.argmax()]
    w_guess = 2*np.abs(freq_final[(np.abs(psd_final - psd_final.max()/2)).argmin()]-f_guess)
    A_guess = np.pi*w_guess*psd_final.max()/2
    guess = [y_guess, f_guess, w_guess, A_guess] #y0,f0,w,A
    # print(guess)
    #fit
    popt, pcov = curve_fit(lorentzian, freq_final,psd_final,
                        p0=guess, bounds=(0,np.inf))
    #print(np.linalg.cond(pcov))
    params = ['offset','resonance freq', 'fwhm', 'area']
    fit_dict = dict(zip(params, popt))
    fit_dict['Q factor'] = fit_dict['resonance freq']/fit_dict['fwhm']
    
    #plot fit
    f_min, f_max = freq_final.min(), freq_final.max()
    f_ext = 0.1*(f_max-f_min)
    freq_fit_range = np.linspace(f_min-f_ext, f_max+f_ext, 100000)
    plt.plot(freq_fit_range,lorentzian(freq_fit_range, *popt))
    fig = fig2html(plt.gcf())
    plt.close()
    
    # print(fit_dict)

    Q = fit_dict['Q factor'] #head_data['Quality factor (Q)']
    k_cant = 2 # N/m
    T = 300 #K
    kb = 1.380649e-23 #J/K
    V_rms = np.sqrt(fit_dict['area'])
    corr_fac = 4/3 #Butt-Jaschke correction for thermal noise
    sens = np.sqrt(corr_fac*kb*T/k_cant)/V_rms/1e-9 #nm/V 

    return sens, k_cant, Q, V_rms, fig