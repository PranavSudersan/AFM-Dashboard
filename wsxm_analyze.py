import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import signal
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import copy

def func_adhesion(force_data, zero_pts):
    segment = 'retract'
    f_zero = force_data['approach']['y'][:zero_pts].mean()
    f_min = force_data[segment]['y'].min()
    return {'value': f_zero - f_min, 'segment': segment, 'zero': f_zero, 'min': f_min}

def func_snapin(defl_data, zero_pts): #CHECK ALGORITHM!
    segment = 'approach'
    # defl_sobel = ndimage.sobel(defl_data[segment]['y']) #sobel transform
    # idx_min = np.argmin(defl_sobel) #id of sharpest corner in defl data
    defl_idx_min = np.argmin(defl_data[segment]['y']) #id of minima of data
    # defl_snapin = defl_data[segment]['y'][idx_min]
    z_snapin = defl_data[segment]['x'][defl_idx_min]
    defl_min = defl_data[segment]['y'][defl_idx_min]
    defl_zero = defl_data[segment]['y'][:zero_pts].mean()
    # z_min = defl_data['approach']['x'][defl_idx_min]
    # print(idx_min)
    return {'value': defl_zero - defl_min, 'segment': segment, 'x': [z_snapin, z_snapin], 'y': [defl_zero, defl_min]}

def func_stiffness(force_data, bad_pts):
    segment = 'approach'
    idx_min = np.argmin(force_data[segment]['y'])
    if idx_min == force_data[segment]['x'].shape[0]-1: #when spectra not good
        return {'value': 0, 'segment': segment, 'x': force_data[segment]['x'][idx_min:], 'y': force_data[segment]['y'][idx_min:]}
    else:
        p, res, rank, sing, rcond = np.polyfit(force_data[segment]['x'][idx_min:], 
                                               force_data[segment]['y'][idx_min:], 1, full=True)
        poly = np.poly1d(p)
        fit_data = {'x': force_data[segment]['x'][idx_min:], 'y': poly(force_data[segment]['x'][idx_min:])}
        return {'value': -p[0], 'segment': segment, 'x': fit_data['x'], 'y': fit_data['y']}

#slope of amplitude change during fd spectroscopy
def func_ampslope(amp_data, range_factor):
    segment = 'approach'
    amp_sobel = ndimage.sobel(amp_data[segment]['y']) #sobel transform
    # ind_max = np.argmax(amp_sobel)
    # sobel_min, sobel_max = amp_sobel.min(), amp_sobel.max()
    # #find range around sobel max to get fit range
    # mid_sobel = sobel_min+(sobel_max-sobel_min)*range_factor
    # ind_mid1 = np.argmin(abs(amp_sobel-mid_sobel))
    # ind_diff = abs(ind_max-ind_mid1)
    # ind_mid2 = ind_max + ind_diff if ind_mid1<ind_max else ind_max - ind_diff 
    # ind_range = [min([ind_mid1, ind_mid2]), max([ind_mid1, ind_mid2])] #indices ordered from small to big

    kmeans = KMeans(n_clusters=2) # Create a KMeans instance with 2 clusters: kmeans
    kmeans.fit(amp_sobel.reshape(-1, 1)) 
    centroids = kmeans.cluster_centers_
    low_cluster, high_cluster = (0, 1) if centroids[0] < centroids[1] else (1, 0) #Get higher value clusters
    labels = kmeans.labels_
    high_cluster_data = amp_sobel[labels == high_cluster]
    high_cluster_indices = np.where(labels.reshape(amp_sobel.shape) == high_cluster)[0]

    # if ind_range[0] < 0 or ind_range[1] > len(amp_data[segment]['x'])-1:
    if len(high_cluster_indices) <= 2: #ignore small clusters
        return {'value': 0, 'segment': segment, 'x': [], 'y': []}
    else:
        p, res, rank, sing, rcond = np.polyfit(amp_data[segment]['x'][high_cluster_indices],
                                               amp_data[segment]['y'][high_cluster_indices], 1, full=True)
        poly = np.poly1d(p)
        fit_data = {'x': amp_data[segment]['x'][high_cluster_indices], 'y': poly(amp_data[segment]['x'][high_cluster_indices])}
        return {'value': -p[0], 'segment': segment, 'x': fit_data['x'], 'y': fit_data['y']}



#TODO: calibration dictionary to get in nm or nN from volts

global FUNC_DICT, CALIB_DICT, SPECT_DICT #CHECK THIS! TODO!

#dictionary of functions defined to extract spectroscopy data properties
#if function outputs other than 'value' is 'x', 'y', set plot type to 'line' below, else, set plot type to
#however it needs to be plotted as a dictionary for each additional output.
#TODO unit calibation add here also where necessary
FUNC_DICT = {'Normal force': {'Adhesion': {'function':func_adhesion,
                                           'kwargs': {'zero_pts': 10},
                                           'plot type': {'zero':'hline', 'min':'hline'}
                                           },
                              'Stiffness': {'function':func_stiffness,
                                            'kwargs': {'bad_pts':1},
                                            'plot type': 'line'
                                            },
                              'Snap-in distance': {'function':func_snapin,
                                                   'kwargs': {'zero_pts': 10},
                                                   'plot type': 'line'
                                                   }
                              },
             'Amplitude': {'Slope-amp':{'function':func_ampslope,
                                        'kwargs': {'range_factor': 0.6},
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
def wsxm_calcspectroparam(spectro_data, channel, unit):
    #perform calculations for parameters (e.g. adhesion, stiffness, check FUNC_DICT) on the single spectroscopy curve
    # spectro_data = data[channel]['curves'][curv_num]['data']
    spectro_data_cali = copy.deepcopy(spectro_data)
    for key in spectro_data_cali.keys(): #calibrate
        spectro_data_cali[key]['y'] = (CALIB_DICT[channel][unit]['factor']*spectro_data_cali[key]['y']) + CALIB_DICT[channel][unit]['offset'] 
    df_spec = convert_spectro2df(spectro_data_cali) #pd.DataFrame.from_dict(data_fd_dict) #for plotting
    # print(channel, unit, CALIB_DICT[channel][unit])
    # df_spec['y'] = (CALIB_DICT[channel][unit]['factor']*df_spec['y']) + CALIB_DICT[channel][unit]['offset'] #calibrate
    data_dict_param = {}
    for param in FUNC_DICT[channel].keys():
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
def calc_spectro_prop(data, channel, img_dir):
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
                    x_pts = int(data[channel][chan_dir]['header']['Number of rows'])
                    y_pts = int(data[channel][chan_dir]['header']['Number of columns'])
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
    sample_rate = float(head_data['Sampling frequency'].split(' ')[0])
    freq_array, z_pow = signal.periodogram(zz, sample_rate, scaling='density') #power spectral density
    z_pow_avg = np.average(z_pow, axis=0) #averaged
    z_pow_max = z_pow_avg.max()
    freq_drive = float(head_data['Resonance frequency'].split(' ')[0])
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