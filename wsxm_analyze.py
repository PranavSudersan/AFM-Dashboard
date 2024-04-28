import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import signal
import scipy.ndimage as ndimage

def func_adhesion(force_data, zero_pts):
    f_zero = force_data['approach']['y'][:zero_pts].mean()
    f_min = force_data['retract']['y'].min()
    return f_zero - f_min, {'zero': f_zero, 'min': f_min}

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
    return defl_zero - defl_min, {'x': [z_snapin, z_snapin], 'y': [defl_zero, defl_min]}

def func_stiffness(force_data, bad_pts):
    segment = 'approach'
    idx_min = np.argmin(force_data[segment]['y'])
    if idx_min == force_data[segment]['x'].shape[0]-1: #when spectra not good
        return 0, {'x': force_data[segment]['x'][idx_min:], 'y': force_data[segment]['y'][idx_min:]}
    else:
        p, res, rank, sing, rcond = np.polyfit(force_data[segment]['x'][idx_min:], 
                                               force_data[segment]['y'][idx_min:], 1, full=True)
        poly = np.poly1d(p)
        fit_data = {'x': force_data[segment]['x'][idx_min:], 'y': poly(force_data[segment]['x'][idx_min:])}
    return -p[0], fit_data

#TODO: calibration dictionary to get in nm or nN from volts

#dictionary of functions defined to extract spectroscopy data properties
# FUNC_DICT = {'Adhesion': {'function':func_adhesion,
#                           'channel': 'Normal force',
#                           'kwargs': {'zero_pts': 10
#                                     }
#                          },
#              'Stiffness': {'function':func_stiffness,
#                            'channel': 'Normal force',
#                            'kwargs': {'bad_pts':1}
#                            },
#              'Snap-in distance': {'function':func_snapin,
#                                   'channel': 'Normal force',
#                                   'kwargs': {}
#                                   },
#              }

FUNC_DICT = {'Normal force': {'Adhesion': {'function':func_adhesion,
                                           'kwargs': {'zero_pts': 10}
                                           },
                              'Stiffness': {'function':func_stiffness,
                                            'kwargs': {'bad_pts':1}
                                            },
                              'Snap-in distance': {'function':func_snapin,
                                                   'kwargs': {'zero_pts': 10}
                                                   }
                              },
             'Amplitude': {},
             'Excitation frequency': {},
             'Phase': {}
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
def wsxm_calcspectroparam(spectro_data, channel):
    #perform calculations for parameters (e.g. adhesion, stiffness, check FUNC_DICT) on the single spectroscopy curve
    # spectro_data = data[channel]['curves'][curv_num]['data']
    df_spec = convert_spectro2df(spectro_data) #pd.DataFrame.from_dict(data_fd_dict) #for plotting
    data_dict_param = {}
    for param in FUNC_DICT[channel].keys():
        # if channel == FUNC_DICT[param]['channel']:
        kwargs = FUNC_DICT[channel][param]['kwargs']
        _, data_dict_param[param] = FUNC_DICT[channel][param]['function'](spectro_data, **kwargs)
        
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
                            param_result,_ = FUNC_DICT[channel][param]['function'](spectro_data, **kwargs)
                            data_dict_param[param][img_dir]['data']['Z'] = np.append(data_dict_param[param][img_dir]['data']['Z'], [param_result])
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
        
