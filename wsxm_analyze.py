import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import signal
# import scipy.ndimage as ndimage
# from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import copy
import re

import spectro_funcs as spf
import fit_funcs as ftf 
from plot_funcs import plotly_lineplot, plotly_heatmap, plotly_dashedlines, fig2html

#TODO: calibration dictionary to get in nm or nN from volts

# global FUNC_DICT, CALIB_DICT, SPECT_DICT #CHECK THIS! TODO!

#dictionary of functions defined to extract spectroscopy data properties
#if function outputs other than 'value' is 'x', 'y', set plot type to 'line' below, else, set plot type to
#however it needs to be plotted as a dictionary for each additional output.
#TODO unit calibation add here also where necessary
FUNC_DICT = {'Normal force': {'Adhesion': {'function': spf.adhesion,
                                           'kwargs': {'method': 'simple',
                                                      'zero_pts': 10,
                                                      'min_percentile': 1,
                                                      'fit_order': 2
                                                     },
                                           'plot type': 'line',#{'zero':'hline', 'min':'hline'}
                                           'unit': '[Normal force]'
                                           },
                              'Stiffness': {'function': spf.stiffness,
                                            'kwargs': {'fit_order':2
                                                      },
                                            'plot type': 'line',
                                            'unit': '[Normal force]/[Z]'
                                            },
                              },
             'Normal deflection': {'Snap-in distance': {'function': spf.snapin,
                                                        'kwargs': {'min_percentile': 1, 
                                                                   'fit_order': 2
                                                                  },
                                                        'plot type': 'line',
                                                        'unit': '[Normal deflection]'
                                                       }
                                  },
             'Amplitude': {'Slope-amp':{'function': spf.ampslope,
                                        'kwargs': {#'range_factor': 0.6,
                                                   'filter_size': 20,
                                                   'method': 'fit', #'fit','average'
                                                   'max_percentile': 99,
                                                   'change': 'up'
                                                  },
                                        'plot type': 'line',
                                        'unit': '[Amplitude]/[Z]'
                                        },
                           'Growth rate':{'function': spf.ampgrowth,
                                          'kwargs': {},
                                          'plot type': 'line',
                                          'unit': '1/[Z]'
                                        }
                          },
             'True Amplitude': {'True Slope-amp':{'function': spf.ampslope,
                                                  'kwargs': {#'range_factor': 0.6,
                                                        'filter_size': 20,
                                                        'method': 'fit', #'fit','average'
                                                        'max_percentile': 99,
                                                        'change': 'down'
                                                  },
                                                  'plot type': 'line',
                                                  'unit': '[True Amplitude]/[Z]'
                                                 },
                                'True Growth rate':{'function': spf.ampgrowth,
                                                    'kwargs': {},
                                                    'plot type': 'line',
                                                    'unit': '1/[Z]'
                                                   }
                               },
             'Excitation frequency': {},
             'Phase': {},
             'True Phase': {}
            }

# calibration dictionary for each channel. ADD MORE CHANNELS!
# add a reference unit as key other than 'factor' and 'offset' in order to get additional calibration
# values for more units.
CALIB_DICT = {'Normal force': {'V': {'factor':1, 'offset':0}, 
                               'nN': {'factor':1, 'offset':0},
                              },
              'Normal deflection': {'V': {'factor':1, 'offset':0}, 
                                    'nm': {'factor':1, 'offset':0}
                                   },
              'Amplitude': {'V': {'factor':1, 'offset':0},
                            'nm': {'factor':1, 'offset':0},
                           },
              'True Amplitude': {'V': {'factor':1, 'offset':0},
                                 'nm': {'factor':1, 'offset':0},
                                },
              'Excitation frequency': {'V': {'factor':1, 'offset':0},
                                       'Hz': {'factor':1, 'offset':0}
                                       },
              'Phase': {'V': {'factor':1, 'offset':0}
                        },
              'True Phase': {'V': {'factor':1, 'offset':0},
                             '°': {'factor':1, 'offset':0}
                            },
              'X': {'Å': {'factor':1, 'offset':0, 'nm': 10},
                    'nm': {'factor':1, 'offset':0, 'nm': 1},
                    'µm': {'factor':1, 'offset':0, 'nm': 0.001}
                   },
              'Y': {'Å': {'factor':1, 'offset':0, 'nm': 10},
                    'nm': {'factor':1, 'offset':0, 'nm': 1},
                    'µm': {'factor':1, 'offset':0, 'nm': 0.001}
                   },
              'Z': {'Å': {'factor':1, 'offset':0, 'nm': 10},
                    'nm': {'factor':1, 'offset':0, 'nm': 1},
                    'µm': {'factor':1, 'offset':0, 'nm': 0.001}
                   },
              'Topography': {'Å': {'factor':1, 'offset':0, 'nm': 10},
                             'nm': {'factor':1, 'offset':0, 'nm': 1},
                             'µm': {'factor':1, 'offset':0, 'nm': 0.001}
                            },
             }

#rename spectroscopy line to standard names: approach and retract
# SPECT_DICT = {'Forward':'approach', 'Backward': 'retract'} 
#rename spectroscopy line to standard names: approach and retract
#also define label name for spectro_data when plotting
SPECT_DICT = {'Forward': 'approach', 'Backward': 'retract',
              'b': 'retract', 'f': 'approach',
              'x': 'Piezo position', 'd': 'Tip-sample distance'
             } 

#update kwargs for FUNCT_DICT
def set_funcdict_kwargs(channel,param,kwargs):
    global FUNC_DICT
    for key, value in kwargs.items():
        FUNC_DICT[channel][param]['kwargs'][key] = value 
        
def set_calibdict_values(channel,unit_kw):
    global CALIB_DICT
    for key, value in unit_kw.items():
        CALIB_DICT[channel][key]['factor'] = value['factor']
        CALIB_DICT[channel][key]['offset'] = value['offset']
        #update all units calibration which are simple multiples of a reference unit
        ref_unit = [k for k in CALIB_DICT[channel][key].keys() if k not in ['factor', 'offset']]
        if len(ref_unit) == 1:
            for key2 in [k for k in CALIB_DICT[channel].keys() if k != key]:
                if ref_unit[0] in CALIB_DICT[channel][key2].keys():
                    CALIB_DICT[channel][key2]['factor'] = value['factor']*(CALIB_DICT[channel][key2][ref_unit[0]]/\
                                                                           CALIB_DICT[channel][key][ref_unit[0]])
                    CALIB_DICT[channel][key2]['offset'] = value['offset']*(CALIB_DICT[channel][key2][ref_unit[0]]/\
                                                                           CALIB_DICT[channel][key][ref_unit[0]])
    # print(channel, CALIB_DICT[channel])

#get correct unit for calculated parameters using exisitng channel calibration units
def parse_paramunit(param, unit_dict, evaluate):
    for key, val in FUNC_DICT.items():
        if param in val.keys():
            param_channel = key
            break
    unit_formula = FUNC_DICT[param_channel][param]['unit']
    pattern = re.compile(r'\[(.*?)\]') # Regular expression to find all occurrences of text within square brackets
    if evaluate == False: #return units
        result = pattern.sub(lambda match: unit_dict.get(match.group(1), match.group(0)), unit_formula)
    else: #return 'factor' corresponding to units from CALIB_DICT for param channel calibration
        result = pattern.sub(lambda match: str(CALIB_DICT.get(match.group(1), 
                                                              match.group(0))[unit_dict.get(match.group(1), 
                                                                                            match.group(0))]['factor']), 
                             unit_formula)
    # def lookup(match): # Function to use as a replacement, which looks up the key in the dictionary
    #     key = match.group(1)  # Extract the key without brackets
    #     return unit_dict.get(key, match.group(0))   # Return the value if key is found, else return the original match
    
    return result
    

# Get spectroscopy data dictionary from force volume data at x,y
# pass unit_dict=None to not make any calibration transform to spectro_data
#pass Normal deflection as defl_data to calculate tip sample distance "d".
#This function is relatively slow, not suitable to get data for whole image, single curves is fine
def wsxm_getspectro(data, channel, img_dir, x=0, y=0, unit_dict=None, calc_d=False):
    # label_dict = {'Forward':'approach', 'Backward': 'retract'} #rename lines
    #initialize spectroscopy data dictionary
    # data_fd_dict = {'x': np.empty(0), 'y': np.empty(0), 'segment': np.empty(0)}
    #spectroscopy keys for the same image direction
    img_keys = [key for key in data[channel].keys() if key.startswith(f'Image {img_dir}')] 
    spectro_data = {}
    for key in img_keys:
        spectro_dir = SPECT_DICT[key.split(' ')[3]]
        img_data = get_imgdata(data[channel][key], channel, unit_dict=unit_dict) #get calibrated data
        x_pt = np.argmin(abs(img_data['X']-x))
        y_pt = np.argmin(abs(img_data['Y']-y))
        # x_pt = np.argmin(abs(data[channel][key]['data']['X']-x)) #NOT CORRECT! UNITS NOT CONSIDERED HERE!
        # y_pt = np.argmin(abs(data[channel][key]['data']['Y']-y))
        # if segment != 'both' and segment != spectro_dir: #skip unwanted segment
        #     continue
        # line_pts = int(data[channel][key]['header']['Number of points per ramp'])
        # data_fd_dict['x'] = np.append(data_fd_dict['x'], data[channel][key]['data']['Z'])
        # data_fd_dict['y'] = np.append(data_fd_dict['y'], data[channel][key]['data']['ZZ'][:,y,x])
        # data_fd_dict['segment'] = np.append(data_fd_dict['segment'], line_pts*[spectro_dir])
        
        # spectro_data[spectro_dir] = {'y': data[channel][key]['data']['ZZ'][:,y_pt,x_pt],
        #                              'x': data[channel][key]['data']['Z']}
        spectro_data[spectro_dir] = {'y': img_data['ZZ'][:,y_pt,x_pt],
                                     'x': img_data['Z']}
        if spectro_dir == 'retract': #flipped array to ensure data starts from highest x (far away) to lowest x (in contact)
            spectro_data[spectro_dir]['x'] = np.flip(spectro_data[spectro_dir]['x'])
            spectro_data[spectro_dir]['y'] = np.flip(spectro_data[spectro_dir]['y'])
    
    # if defl_data != None and unit_dict != None:
    #     spectro_data = calc_tipsampledistance(spectro_data, defl_data)
    
    if calc_d == True: #calculate tip sample distance and insert it as 'd' in spectro_data
        defl_data = wsxm_getspectro(data, 'Normal deflection', img_dir, x=x, y=y, unit_dict=unit_dict, calc_d=False)
        spectro_data = calc_tipsampledistance(spectro_data, defl_data)
    
    # data_fd = pd.DataFrame.from_dict(data_fd_dict)
    # #perform calculations for parameters (e.g. adhesion, stiffness, check FUNC_DICT) on the single spectroscopy curve
    # data_dict_param = {}
    # for param in FUNC_DICT[channel].keys():
    #     kwargs = FUNC_DICT[channel][param]['kwargs']
    #     _, data_dict_param[param] = FUNC_DICT[channel][param]['function'](spectro_data, **kwargs)   
        
    # print(x,y,data['ZZ'])
    return spectro_data

# Convert spectroscopy data dictionary to dataframe for plotting and calculate parameter
# pass unit_dict=None to not make any calibration transform to spectro_data. BE CAREFUL when used in conjunction with wsxm_getspectro,
#unit transformation might be done twice, make sure to pass unit_dict=None if unit operation already done earlier (eg. in wsxm_getspectro)
#pass Normal deflection as defl_data to calculate tip sample distance "d"
def wsxm_calcspectroparam(spectro_data, channel, unit_dict=None, calc_params=True, properties=[], defl_data=None):
    #perform calculations for parameters (e.g. adhesion, stiffness, check FUNC_DICT) on the single spectroscopy curve
    # spectro_data = data[channel]['curves'][curv_num]['data']
    spectro_data_cali = copy.deepcopy(spectro_data)
    
    if unit_dict != None: #return spectro data without calibration
        for key in spectro_data_cali.keys(): #calibrate
            spectro_data_cali[key]['y'] = (CALIB_DICT[channel][unit_dict[channel]]['factor']*spectro_data_cali[key]['y']) + \
                                            CALIB_DICT[channel][unit_dict[channel]]['offset'] 
            spectro_data_cali[key]['x'] = (CALIB_DICT['Z'][unit_dict['Z']]['factor']*spectro_data_cali[key]['x']) + \
                                            CALIB_DICT['Z'][unit_dict['Z']]['offset'] #CHECK THIS! THIS ONLY APPLIES FOR SPECTROSCOPY IN Z
        if defl_data != None:
            spectro_data_cali = calc_tipsampledistance(spectro_data_cali, defl_data)
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

#calculate tip sample distance data using normal deflection data and include into spectro_data as 'd' channel.
#here defl_data is a dictionary similar to spectro_data corresponding to Normal deflection channel
def calc_tipsampledistance(spectro_data, defl_data):
    for key in spectro_data.keys():
        xini_ind = np.argmax(spectro_data[key]['x']) #position furthest from sample
        kwargs = FUNC_DICT['Normal deflection']['Snap-in distance']['kwargs']
        spectro_data[key]['d'] = spectro_data[key]['x'] + (defl_data[key]['y']-defl_data[key]['y'][xini_ind]) #tip sample distance
        snapin_output = spf.snapin(defl_data, **kwargs)
        if 'index' in snapin_output.keys(): #shift x data such that point of snap-in is taken as zero tip sample distance
            # print('d', spectro_data[key]['d'][snapin_output['index']], defl_data[key]['y'][xini_ind], xini_ind, snapin_output['index'],
            #      spectro_data[key]['x'][snapin_output['index']])
            spectro_data[key]['d'] =  spectro_data[key]['d'] - spectro_data[key]['d'][snapin_output['index']]
    return spectro_data
                        
#convert spectro data dictionary to dataframe for plotting
def convert_spectro2df(data_dict):
    data = {}
    for k in list(data_dict.values())[0].keys(): #initialize x,y,d if it exists in data_dict
        data[k] = np.empty(0)
    data['segment'] = np.empty(0)
    # data = {'x': np.empty(0), 'y': np.empty(0), 'segment': np.empty(0)}
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
                                    #flipped array to ensure data starts from highest x (far away) to lowest x (in contact)
                                    if spectro_dir == 'retract':
                                        spectro_data[spectro_dir]['x'] = np.flip(spectro_data[spectro_dir]['x'])
                                        spectro_data[spectro_dir]['y'] = np.flip(spectro_data[spectro_dir]['y'])
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
def get_imgdata(data_dict_chan, channel, x=None, y=None, z=None, unit_dict=None):
                # unit_xyz={'X':'nm', 'Y':'nm', 'Z':'nm'}):
    # data0 = data_dict_chan['data'][style[0]]
    # data1 = data_dict_chan['data'][style[1]]
    # data_mat = np.meshgrid(data0, data1)
    img_data = {}
    # print(CALIB_DICT['X'])
    if 'ZZ' in data_dict_chan['data'].keys(): #for force volume data
        if unit_dict == None:
            img_data['X'] = data_dict_chan['data']['X']
            img_data['Y'] = data_dict_chan['data']['Y']
            img_data['Z'] = data_dict_chan['data']['Z']
        else:
            img_data['X'] = (CALIB_DICT['X'][unit_dict['X']]['factor']*data_dict_chan['data']['X']) + CALIB_DICT['X'][unit_dict['X']]['offset'] 
            img_data['Y'] = (CALIB_DICT['Y'][unit_dict['Y']]['factor']*data_dict_chan['data']['Y']) + CALIB_DICT['Y'][unit_dict['Y']]['offset']
            img_data['Z'] = (CALIB_DICT['Z'][unit_dict['Z']]['factor']*data_dict_chan['data']['Z']) + CALIB_DICT['Z'][unit_dict['Z']]['offset'] 
        # if style == 'XY':
        if z != None:
            z_pt = np.argmin(abs(img_data['Z']-z))
            unit = 'V' if unit_dict==None else unit_dict[channel]
            img_data['XY'] = (CALIB_DICT[channel][unit]['factor']*data_dict_chan['data']['ZZ'][z_pt,:,:]) + \
                                CALIB_DICT[channel][unit]['offset'] 
            # z_pt = np.argmin(abs(data_dict_chan['data']['Z']-z))
            # img_data = data_dict_chan['data']['ZZ'][z_pt,:,:] #1st index:xy sections, 2nd index:xz sections, 3rd index: yz sections
        # elif style == 'XZ':
        if y != None:
            y_pt = np.argmin(abs(img_data['Y']-y))
            unit = 'V' if unit_dict==None else unit_dict[channel]
            # print(y_pt)
            img_data['XZ'] = (CALIB_DICT[channel][unit]['factor']*data_dict_chan['data']['ZZ'][:,y_pt,:]) + \
                                CALIB_DICT[channel][unit]['offset']
            # y_pt = np.argmin(abs(data_dict_chan['data']['Y']-y))
            # img_data = data_dict_chan['data']['ZZ'][:,y_pt,:]
        # elif style == 'YZ':
        if x != None:
            x_pt = np.argmin(abs(img_data['X']-x))
            unit = 'V' if unit_dict==None else unit_dict[channel]
            img_data['YZ'] = (CALIB_DICT[channel][unit]['factor']*data_dict_chan['data']['ZZ'][:,:,x_pt]) + \
                                CALIB_DICT[channel][unit]['offset']
            # x_pt = np.argmin(abs(data_dict_chan['data']['X']-x))
            # img_data = data_dict_chan['data']['ZZ'][:,:,x_pt]
        if x == None and y == None and z == None:
            unit = 'V' if unit_dict==None else unit_dict[channel] #return orignal cube data after applying caliration
            img_data['ZZ'] = (CALIB_DICT[channel][unit]['factor']*data_dict_chan['data']['ZZ']) + \
                                CALIB_DICT[channel][unit]['offset']
    else: #for usual image data
        if unit_dict == None:
            img_data['X'] = data_dict_chan['data']['X']
            img_data['Y'] = data_dict_chan['data']['Y']
            img_data['Z'] = data_dict_chan['data']['Z']
        else:
            img_data['X'] = (CALIB_DICT['X'][unit_dict['X']]['factor']*data_dict_chan['data']['X']) + CALIB_DICT['X'][unit_dict['X']]['offset'] 
            img_data['Y'] = (CALIB_DICT['Y'][unit_dict['Y']]['factor']*data_dict_chan['data']['Y']) + CALIB_DICT['Y'][unit_dict['Y']]['offset']
            if channel in CALIB_DICT.keys():
                img_data['Z'] = (CALIB_DICT[channel][unit_dict[channel]]['factor']*data_dict_chan['data']['Z']) + \
                                    CALIB_DICT[channel][unit_dict[channel]]['offset']
            else: #multiply parameter channel by calibration factors. NOTE! OFFSETS ARE IGNORED HERE! MAY NOT BE CORRECT TO CALIBRATE PARAMS
                img_data['Z'] = data_dict_chan['data']['Z']*eval(parse_paramunit(channel, unit_dict, evaluate=True))
            
        # img_data = data_dict_chan['data']['Z']
    # data_mat.append(img_data)
    # return data_mat[0], data_mat[1], data_mat[2]
    return img_data

#get data at a specific line of an image. x=vertical line, y=horizontal line
#pass unit_dict=None for no calibration transform on data
def get_imgline(data_dict_chan, channel, x=None, y=None, unit_dict=None):
                # unit_xy={'X':'nm', 'Y':'nm'}):
    img_data = {}
    if unit_dict == None:
        img_data['X'] = data_dict_chan['data']['X']
        img_data['Y'] = data_dict_chan['data']['Y']
    else:
        img_data['X'] = (CALIB_DICT['X'][unit_dict['X']]['factor']*data_dict_chan['data']['X']) + CALIB_DICT['X'][unit_dict['X']]['offset'] 
        img_data['Y'] = (CALIB_DICT['Y'][unit_dict['Y']]['factor']*data_dict_chan['data']['Y']) + CALIB_DICT['Y'][unit_dict['Y']]['offset'] 
    if x != None:
        x_pt = np.argmin(abs(img_data['X']-x))
        if unit_dict == None:
            img_data['Z'] = data_dict_chan['data']['Z'][:,x_pt]
        else:
            if channel in CALIB_DICT.keys():
                img_data['Z'] = (CALIB_DICT[channel][unit_dict[channel]]['factor']*data_dict_chan['data']['Z'][:,x_pt]) + \
                                    CALIB_DICT[channel][unit_dict[channel]]['offset'] 
            else: #multiply parameter channel by calibration factors. NOTE! OFFSETS ARE IGNORED HERE! MAY NOT BE CORRECT TO CALIBRATE PARAMS
                img_data['Z'] = data_dict_chan['data']['Z'][:,x_pt]*eval(parse_paramunit(channel, unit_dict, evaluate=True))
        # x_pt = np.argmin(abs(data_dict_chan['data']['X']-x))
        return img_data['Y'], img_data['Z']#[:,x_pt]
    if y != None:
        # unit = 'V' if unit_dict==None else unit_dict[channel]
        y_pt = np.argmin(abs(img_data['Y']-y))
        if unit_dict == None:
            img_data['Z'] = data_dict_chan['data']['Z'][y_pt,:]
        else:
            if channel in CALIB_DICT.keys():
                img_data['Z'] = (CALIB_DICT[channel][unit_dict[channel]]['factor']*data_dict_chan['data']['Z'][y_pt,:]) + \
                                    CALIB_DICT[channel][unit_dict[channel]]['offset'] 
            else: #multiply parameter channel by calibration factors. NOTE! OFFSETS ARE IGNORED HERE! MAY NOT BE CORRECT TO CALIBRATE PARAMS
                img_data['Z'] = data_dict_chan['data']['Z'][y_pt,:]*eval(parse_paramunit(channel, unit_dict, evaluate=True))
        # y_pt = np.argmin(abs(data_dict_chan['data']['Y']-y))
        return img_data['X'], img_data['Z']#[y_pt,:]
                
def combine_forcevol_data(data, channel_list, label_data=[]):
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
        
        if len(label_data) != 0:
            output_data['label'] = []
            # label_reshaped = np.dstack([label_data]*z_len).flatten(order='C')
            label_reshaped = np.repeat(label_data.flatten('F'), z_len)
            for spec_dir in ['Forward', 'Backward']:
                output_data['label'].append(label_reshaped)
            output_data['label'] = np.concatenate(output_data['label'])
        
        # print(label_cube.shape, z_len, label_data.shape, label_cube.flatten(order='F').shape)    
        for chan in channel_list:
            output_data[chan] = []
            for spec_dir in ['Forward', 'Backward']:
                output_data[chan].append(data[chan][f'Image {img_dir} with {spec_dir} Ramps']['data']['ZZ'].flatten(order='F'))
                # print(len(output_data[chan]), output_data[chan][-1].shape,  len(output_data['segment']), len(output_data['segment'][-1]))
            output_data[chan] = np.concatenate(output_data[chan])
        
        # print(len(output_data['label']), len(output_data['Z']), len(output_data['Normal force']))
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

def get_psd_calib(amp_data):
    zz_list = []
    for img_dir in amp_data.keys():
        head_data = amp_data[img_dir]['header']
        zz_amp = amp_data[img_dir]['data']['Z']
        # xx, yy, zz_amp = get_imgdata(amp_data[img_dir])

        # xx, yy, zz_phase= get_imgdata(phase_data[img_dir])

        #true amplitude calculated from amp and phase channels
        # zz_i = np.sqrt(np.square(zz_amp) + np.square(zz_phase))
        zz_list.append(zz_amp)
    
    zz = np.concatenate(zz_list, axis=1)
    
    # if img_dir == 'Backward':
    #     print(img_dir)
    #     zz = np.flip(zz, axis=1)
    
    # plt.pcolormesh(zz, cmap='afmhot')    
    # plt.colorbar()
    fig = fig2html(plotly_heatmap(z_mat=zz, x=None, y=None), plot_type='plotly')
    # plt.close()
    
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


def get_calib(df_on, df_off, ind):
    freq_final = df_on['frequency'].iloc[ind]
    psd_final = df_on['psd'].iloc[ind] - df_off['psd'].iloc[ind]
    # plt.plot(freq_final, psd_final, 'r')
    # plt.plot(freq_final, df_on['psd'].iloc[ind], 'y', alpha=0.5)
    # plt.plot(freq_final, df_off['psd'].iloc[ind], 'y', alpha=0.5)
    #plt.show()
    psd_data = pd.DataFrame({'Frequency': freq_final, 
                             'final': psd_final, 
                             'laser on': df_on['psd'].iloc[ind],
                             'laser off': df_off['psd'].iloc[ind]
                            })
    psd_df_long = pd.melt(psd_data, id_vars=['Frequency'], value_vars=['final', 'laser on', 'laser off'],
                         var_name='name', value_name='PSD')
    
    
    #guess = [0, 76000, 2000, 100000]
    y_guess = 0 #psd_final.min()
    f_guess = freq_final[psd_final.argmax()]
    w_guess = 2*np.abs(freq_final[(np.abs(psd_final - psd_final.max()/2)).argmin()]-f_guess)
    A_guess = np.pi*w_guess*psd_final.max()/2
    guess = [y_guess, f_guess, w_guess, A_guess] #y0,f0,w,A
    # print(guess)
    #fit
    popt, pcov = curve_fit(ftf.lorentzian, freq_final,psd_final,
                        p0=guess, bounds=(0,np.inf))
    #print(np.linalg.cond(pcov))
    params = ['offset','resonance freq', 'fwhm', 'area']
    fit_dict = dict(zip(params, popt))
    fit_dict['Q factor'] = fit_dict['resonance freq']/fit_dict['fwhm']
    
    #plot fit
    f_min, f_max = freq_final.min(), freq_final.max()
    f_ext = 0.1*(f_max-f_min)
    fit_n = 8000
    freq_fit_range = np.linspace(f_min-f_ext, f_max+f_ext, fit_n)
    # plt.plot(freq_fit_range,ftf.lorentzian(freq_fit_range, *popt), 'k--')
    # psd_df_fit = pd.DataFrame({'Frequency': freq_fit_range, 'name': ['fit']*fit_n, 
    #                           'value': ftf.lorentzian(freq_fit_range, *popt)})
    # psd_df_all = pd.concat([psd_df_long, psd_df_fit])
    
    fig = plotly_lineplot(data=psd_df_long, x="Frequency", y="PSD", color="name",
                         color_discrete_sequence=['red', 'blue', 'skyblue'])#, 'yellow'])
    plotly_dashedlines(plot_type='line',fig=fig, x=freq_fit_range, 
                       y=ftf.lorentzian(freq_fit_range, *popt), line_width=2, name='fit')
    fig.update_layout(legend_title=None)
    # fig.data[-1].line.dash = 'dash'
    # plt.close()
    fig_html = fig2html(fig, plot_type='plotly')
    # print(fit_dict)

    Q = fit_dict['Q factor'] #head_data['Quality factor (Q)']
    k_cant = 2 # N/m
    T = 300 #K
    kb = 1.380649e-23 #J/K
    V_rms = np.sqrt(fit_dict['area'])
    corr_fac = 4/3 #Butt-Jaschke correction for thermal noise
    sens = np.sqrt(corr_fac*kb*T/k_cant)/V_rms/1e-9 #nm/V 

    return sens, k_cant, Q, V_rms, fig_html