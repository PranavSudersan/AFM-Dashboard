import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
from scipy.special import kv
# import scipy.ndimage as ndimage
# from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import copy
import re

from Sader_GCI_demo import SaderGCI_CalculateK
import spectro_funcs as spf
import fit_funcs as ftf 
import transform_funcs as tsf
from plot_funcs import plotly_lineplot, plotly_heatmap, plotly_dashedlines, seaborn_lineplot, matplotlib_dashedlines, fig2html

#TODO: calibration dictionary to get in nm or nN from volts

# global FUNC_DICT, CALIB_DICT, SPECT_DICT #CHECK THIS! TODO!

#dictionary of functions defined to extract spectroscopy data properties
#if function outputs other than 'value' is 'x', 'y', set plot type to 'line' below, else, set plot type to
#however it needs to be plotted as a dictionary for each additional output.
#IMPORTANT: Channels are reported in this order such than parameters from subsequent channels can use results from
#the previous channel. Use set_funcdict_kwargs inside spectro_funcs to update the corresponding kwargs of the parameters.
#Also make sure to process channel data in the same order as FUNC_DICT
FUNC_DICT = {'Normal deflection': {'Snap-in distance': {'function': spf.snapin,
                                                        'kwargs': {
                                                            'segment': 'approach',
                                                            'method': 'gradient', #'gradient', 'minima'
                                                                   'min_percentile': 1, 
                                                                   'fit_order': 2,
                                                                   'back_pts': 10,
                                                                   'findmax': True,
                                                                   'zero': 'max' #'max', 'mean', 'median', 'ini', 'driifted'
                                                                  },
                                                        'plot type': 'line',
                                                        'unit': '[Normal deflection]'
                                                       }
                                  },
             'Normal force': {'Adhesion': {'function': spf.adhesion,
                                           'kwargs': {
                                               'segment': 'retract',
                                               'method': 'simple',
                                                      'zero_pts': 10,
                                                      'min_percentile': 1,
                                                      'fit_order': 2
                                                     },
                                           'plot type': 'line',#{'zero':'hline', 'min':'hline'}
                                           'unit': '[Normal force]'
                                           },
                              'Stiffness': {'function': spf.stiffness,
                                            'kwargs': {
                                                'segment': 'approach',
                                                'method': 'best gradient', #'best gradient', 'simple poly'
                                                       'fit_order':1, #1,2 for "best gradient" method
                                                       'snapin_index': None, #updated after running spf.snapin above
                                                       'percentile_range': (0, 50),
                                                       'filter_size': 30
                                                      },
                                            'plot type': 'line',
                                            'unit': '[Normal force]/[Z]'
                                            },
                              },
             'Amplitude': {'Slope-amp':{'function': spf.ampslope,
                                        'kwargs': {
                                            'segment': 'approach',
                                            'filter_size': 20,
                                                   'method': 'fit', #'fit','average'
                                                   'max_percentile': 99,
                                                   'change': 'up',
                                                   'num_pts': 10, 
                                                   'change_factor': 20
                                                  },
                                        'plot type': 'line',
                                        'unit': '[Amplitude]/[Z]'
                                        },
                           'Growth rate':{'function': spf.ampgrowth,
                                          'kwargs': {
                                              'segment': 'approach',
                                              'change': 'up'},
                                          'plot type': 'line',
                                          'unit': '1/[Z]'
                                        }
                          },
             'True Amplitude': {'True Slope-amp':{'function': spf.ampslope,
                                                  'kwargs': {
                                                      'segment': 'approach',
                                                      'filter_size': 20,
                                                             'method': 'fit', #'fit','average'
                                                             'max_percentile': 99,
                                                             'change': 'down',
                                                             'num_pts': 10, 
                                                             'change_factor': 20
                                                            },
                                                  'plot type': 'line',
                                                  'unit': '[True Amplitude]/[Z]'
                                                 },
                                'True Growth rate':{'function': spf.ampgrowth,
                                                    'kwargs': {'segment': 'approach','change': 'down'},
                                                    'plot type': 'line',
                                                    'unit': '1/[Z]'
                                                   }
                               },
             'Excitation frequency': {},
             'Phase': {},
             'True Phase': {},
             'Amplitude-sample distance': {},
             'Tip position': {},
             'Sample deformation': {},
             'X-Y components': {},
             'Amplitude dissipated': {},
             'Energy dissipated': {},
             'Frequency shift': {},
             '2nd Feedback Out': {},
             'Lateral force': {},
            }

# calibration dictionary for each channel. ADD MORE CHANNELS!
# add a reference unit as key other than 'factor' and 'offset' in order to get additional calibration
# values for more units. The value refresents the value of the reference unit relative to the actual unit.
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
                                       'Hz': {'factor':1, 'offset':0, 'Hz': 1},
                                       'kHz': {'factor':1, 'offset':0, 'Hz': 0.001}
                                       },
              'Phase': {'V': {'factor':1, 'offset':0}},
              'True Phase': {'rad': {'factor':1, 'offset':0,},
                             '°': {'factor':180/np.pi, 'offset':0}
                            },
              'Amplitude-sample distance': {'V': {'factor':1, 'offset':0}, 
                                            'nm': {'factor':1, 'offset':0}
                                            },
              'Tip position': {'V': {'factor':1, 'offset':0}, 
                                            'nm': {'factor':1, 'offset':0}
                                            },
              'Sample deformation': {'V': {'factor':1, 'offset':0}, 
                                     'nm': {'factor':1, 'offset':0}
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
              'Spring constant': {'N/m': {'factor':1, 'offset':0}},
              'Resonance frequency': {'Hz': {'factor':1, 'offset':0}},
              'Quality factor': {'': {'factor':1, 'offset':0}},
              'X-Y components': {'V': {'factor':1, 'offset':0}},
              'Amplitude dissipated': {'V': {'factor':1, 'offset':0},
                                       'nm': {'factor':1, 'offset':0}
                                      },
              'Energy dissipated': {'V2': {'factor':1, 'offset':0},
                                    'aJ': {'factor':1, 'offset':0} #attoJoules (10^-18)
                                   },
              'Frequency shift': {'V': {'factor':1, 'offset':0},
                                  'Hz': {'factor':1, 'offset':0, 'Hz': 1},
                                  'kHz': {'factor':1, 'offset':0, 'Hz': 0.001}
                                  },
              '2nd Feedback Out': {'V': {'factor':1, 'offset':0}},
              'Lateral force': {'V': {'factor':1, 'offset':0}},
             }

#rename spectroscopy line to standard names: approach and retract
# SPECT_DICT = {'Forward':'approach', 'Backward': 'retract'} 
#rename spectroscopy line to standard names: approach and retract
#also define label name for spectro_data when plotting
SPECT_DICT = {'Forward': 'approach', 'Backward': 'retract',
              'b': 'retract', 'f': 'approach',
              'x': 'Piezo position', 'd': 'Tip-sample distance',
              'z': 'Piezo position shifted'
             } 

#update kwargs for FUNCT_DICT
def set_funcdict_kwargs(channel,param,kwargs):
    global FUNC_DICT
    for key, value in kwargs.items():
        FUNC_DICT[channel][param]['kwargs'][key] = value 

#update kwargs for FUNCT_DICT
def get_funcdict_kwargs(channel,param):
    global FUNC_DICT
    return FUNC_DICT[channel][param]['kwargs']
        
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
    
def get_calibdict_value(channel,unit):
    global CALIB_DICT
    return CALIB_DICT[channel][unit]

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
        spectro_data = calc_tipsampledistance(spectro_data, defl_data, channel)
    
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
#pass Normal deflection as defl_data (without calibration) to calculate tip sample distance "d"
def wsxm_calcspectroparam(spectro_data, channel, unit_dict=None, calc_params=True, properties=[], defl_data=None):
    global CALIB_DICT
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
            defl_data_cali = copy.deepcopy(defl_data)
            for key in defl_data_cali.keys(): #calibrate defl_data first    
                defl_data_cali[key]['y'] = (CALIB_DICT['Normal deflection'][unit_dict['Normal deflection']]['factor']*defl_data_cali[key]['y']) + \
                                                CALIB_DICT['Normal deflection'][unit_dict['Normal deflection']]['offset'] 
                defl_data_cali[key]['x'] = (CALIB_DICT['Z'][unit_dict['Z']]['factor']*defl_data_cali[key]['x']) + \
                                                CALIB_DICT['Z'][unit_dict['Z']]['offset']
            spectro_data_cali = calc_tipsampledistance(spectro_data_cali, defl_data_cali, channel)
            # print(spectro_data_cali['approach']['x'], spectro_data_cali['approach']['d'], defl_data_cali['approach']['y'])
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
#additionally, also calculated 'amplitude_sample distance" and "sample deformation" data using tip-sample distance data
def calc_tipsampledistance(spectro_data, defl_data, channel):
    for key in spectro_data.keys():
        # xini_ind = np.argmax(spectro_data[key]['x']) #position furthest from sample
        kwargs = FUNC_DICT['Normal deflection']['Snap-in distance']['kwargs']
        # print(kwargs)
        snapin_output = spf.snapin(defl_data, **kwargs)
        kwargs = FUNC_DICT['Normal force']['Stiffness']['kwargs']
        # print(kwargs)
        stiffness_output = spf.stiffness(defl_data, **kwargs)
        # spectro_data[key]['d'] = spectro_data[key]['x'] + defl_data[key]['y']-snapin_output['zero'] #tip sample distance
        spectro_data[key]['d'] = spectro_data[key]['x'] + defl_data[key]['y']
        spectro_data[key]['d'] = spectro_data[key]['d'] - np.average(spectro_data[key]['d'][stiffness_output['fit_index']]) #tip sample distance
        # print(np.average(spectro_data[key]['d'][stiffness_output['fit_index']]), stiffness_output['fit_index'], key)
        if 'index_min' in snapin_output.keys(): #shift x data such that point of snap-in is taken as zero tip sample distance
            # print('d', spectro_data[key]['d'][snapin_output['index']], defl_data[key]['y'][xini_ind], xini_ind, snapin_output['index'],
            #      spectro_data[key]['x'][snapin_output['index']])
            # spectro_data[key]['d'] =  spectro_data[key]['d'] - spectro_data[key]['d'][snapin_output['index_min']]
            # spectro_data[key]['z'] =  spectro_data[key]['x'] - spectro_data[key]['x'][snapin_output['index_surf']] #shifted piezo distance
            spectro_data[key]['z'] =  spectro_data[key]['x'] - stiffness_output['x_surf'] #shifted piezo distance
            if channel == 'Sample deformation':
                spectro_data[key]['y'] = -spectro_data[key]['d'] 
                if key == 'approach':
                    spectro_data[key]['y'][:snapin_output['index_min']] = 0
                elif key == 'retract':
                    kwargs2 = FUNC_DICT['Normal force']['Adhesion']['kwargs']
                    adh_output = spf.adhesion(defl_data, **kwargs2)
                    spectro_data[key]['y'][:adh_output['index']] = 0
                    
        if channel == 'Amplitude-sample distance':
            spectro_data[key]['y'] = spectro_data[key]['d'] - np.absolute(spectro_data[key]['y'])
        if channel == 'Tip position':
            spectro_data[key]['y'] = spectro_data[key]['z'] + defl_data[key]['y']-snapin_output['zero']
            # np.set_printoptions(threshold=np.inf)
            # print(spectro_data[key]['d'], defl_data[key]['y'], snapin_output['zero'], defl_data[key]['y']-snapin_output['zero'], spectro_data[key]['y'])
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
    # for chan_dir in data['Topography'].keys():
    for img_dir in ['Forward', 'Backward']:
        chan_dir = [key for key in data['Topography'].keys() if key.startswith(f'Image {img_dir}')][0] #dummy channel direction to get X,Y data
        # img_dir = chan_dir.split(' ')[1] #'Forward' or 'Backward'
        x_pts = int(data['Topography'][chan_dir]['header']['Number of rows [General Info]'])
        y_pts = int(data['Topography'][chan_dir]['header']['Number of columns [General Info]'])
        for y in range(y_pts): #calculated all properties point by point in x,y, since properties can be related to each other for the same x,y
            for x in range(x_pts):
                for channel in FUNC_DICT.keys(): #following the order of channels in FUNC_DICT
                    if channel in chans:
                # for channel in chans:
                        params = FUNC_DICT[channel].keys()
                        for param in params:
                            if param in properties:
                                if param not in data_dict_param.keys():
                                    data_dict_param[param] = {}
                                # for chan_dir in data[channel].keys():
                                    # img_dir = chan_dir.split(' ')[1] #'Forward' or 'Backward'
                                if img_dir not in data_dict_param[param].keys():
                                    data_dict_param[param][img_dir] = {'data': {'X': data[channel][chan_dir]['data']['X'], 
                                                                                'Y':  data[channel][chan_dir]['data']['Y'], 
                                                                                'Z': np.empty(0)},
                                                                       'header':{}
                                                                      }

                            #get common data from first channel e.g. number of points
                            # channel = FUNC_DICT[params[0]]['channel']
                            # key = list(data[channel].keys())[0]
                                    # x_pts = int(data[channel][chan_dir]['header']['Number of rows [General Info]'])
                                    # y_pts = int(data[channel][chan_dir]['header']['Number of columns [General Info]'])
                                    # for y in range(y_pts):
                                    #     for x in range(x_pts):
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
                            # print(f'{param} calculated')
                    # data_dict_spectro[spectro_dir] = adh_data.reshape(x_pts,y_pts)
    for param in data_dict_param.keys():
        for img_dir in data_dict_param[param].keys():
            data_dict_param[param][img_dir]['data']['Z'] = data_dict_param[param][img_dir]['data']['Z'].reshape(x_pts, y_pts)
        print(f'{param} calculated')
    
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
                
def combine_forcevol_data(data, channel_list, label_data=[], unit_dict=None):
    output_all_dict = {}
    for img_dir in ['Forward', 'Backward']:
        output_data = {}
        img_data_ini = get_imgdata(data[channel_list[0]][f'Image {img_dir} with Forward Ramps'], channel_list[0],
                                   unit_dict=unit_dict)
        z_data_ini = img_data_ini['Z']
        # z_data_temp = data[channel_list[0]][f'Image {img_dir} with Forward Ramps']['data']['Z']

        # z_data_full = np.concatenate([z_data, z_data]) #for both approach and retract for all channels
        x_len =  len(img_data_ini['X']) #len(data[channel_list[0]][f'Image {img_dir} with Forward Ramps']['data']['X'])
        y_len =  len(img_data_ini['Y']) #len(data[channel_list[0]][f'Image {img_dir} with Forward Ramps']['data']['Y'])
        z_len = len(z_data_ini)
        # output_data['Z'] = np.reshape([[z_data_full]*(x_len*y_len)], (x_len,y_len,len(z_data_full))).flatten()
        specdir_list = []
        z_list = []
        z_array_dict = {'Forward': z_data_ini, 'Backward': np.flip(z_data_ini)}
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
                img_data_i = get_imgdata(data[chan][f'Image {img_dir} with {spec_dir} Ramps'], chan, unit_dict=unit_dict)
                output_data[chan].append(img_data_i['ZZ'].flatten(order='F'))
                # output_data[chan].append(data[chan][f'Image {img_dir} with {spec_dir} Ramps']['data']['ZZ'].flatten(order='F'))
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

def get_psd_calib(amp_data, phase_data):
    zz_amp_list = []
    zz_phase_list = []
    for img_dir in ['Forward', 'Backward']:#amp_data.keys():
        if img_dir in amp_data.keys():
            head_data = amp_data[img_dir]['header']
            # z_data_i = tsf.flatten_line(data_dict_chan_i['data'], order=1)
            # zz_amp = amp_data[img_dir]['data']['Z']
            # zz_phase = phase_data[img_dir]['data']['Z']
            zz_amp = tsf.flatten_line(amp_data[img_dir]['data'], order=1, pos_shift=False)
            zz_phase = tsf.flatten_line(phase_data[img_dir]['data'], order=1, pos_shift=False)
            # xx, yy, zz_amp = get_imgdata(amp_data[img_dir])
    
            # xx, yy, zz_phase= get_imgdata(phase_data[img_dir])
            if img_dir == 'Forward':
                zz_amp = np.flip(zz_amp, axis=1)
                zz_phase = np.flip(zz_phase, axis=1)
            #true amplitude calculated from amp and phase channels
            # zz_i = np.sqrt(np.square(zz_amp) + np.square(zz_phase))
            zz_amp_list.append(zz_amp)
            zz_phase_list.append(zz_phase)
    
    zz_amp_full = np.concatenate(zz_amp_list, axis=1)
    zz_phase_full = np.concatenate(zz_phase_list, axis=1)
    
    # if img_dir == 'Backward':
    #     print(img_dir)
    #     zz = np.flip(zz, axis=1)
    
    # plt.pcolormesh(zz, cmap='afmhot')    
    # plt.colorbar()
    # fig = fig2html(plotly_heatmap(z_mat=zz_amp_full, x=None, y=None), plot_type='plotly')
    fig = fig2html(zz_amp_full, plot_type='image')
    # plt.close()
    
    #Obtain Power Spectral Density of data
    #sample_rate = 2*num_pts*float(head_data['X-Frequency'].split(' ')[0])
    sample_rate = float(head_data['Sampling frequency [Miscellaneous]'].split(' ')[0])
    freq_array, z_pow_amp = signal.periodogram(zz_amp_full, sample_rate, scaling='density') #power spectral density
    freq_array, z_pow_phase = signal.periodogram(zz_phase_full, sample_rate, scaling='density') #power spectral density
    z_pow_true = z_pow_amp + z_pow_phase #total power corresponding to "true amplitude"
    z_pow_avg = np.average(z_pow_true, axis=0) #averaged
    z_pow_max = z_pow_avg.max()
    freq_drive = float(head_data['Resonance frequency [Dynamic settings]'].split(' ')[0])
    freq_array_shifted = freq_array + freq_drive
    # plt.plot(freq_array, z_pow_avg)
    # plt.show()

    z = zz_amp_full.flatten()
    # z_rms = np.sqrt(2)*np.sqrt(np.mean(np.square(z))) #rms of true amplitude assuming both "amp' and 'phase" channels are half of power each
    #true amplitude calculated from amp and phase channels
    z_rms = np.sqrt(np.mean(np.square(zz_amp_full.flatten()) + np.square(zz_phase_full.flatten())))
    # print(zz.min(), zz.max(), z_rms)
    return freq_array_shifted, z_pow_avg, z_pow_max, z_rms, fig


def get_calib(df_on, df_off, T, lever_number, userid, password, datarange=(0,1)):
    freq_raw = df_on['frequency']
    psd_raw = df_on['psd'] - df_off['psd']
    #only take a small window of data (if psd is bad)
    ind0, ind1 = int(datarange[0]*len(psd_raw)), int(datarange[1]*len(psd_raw))
    psd_final = psd_raw[ind0:ind1]
    freq_final = freq_raw[ind0:ind1]
    # plt.plot(freq_final, psd_final, 'r')
    # plt.plot(freq_final, df_on['psd'].iloc[ind], 'y', alpha=0.5)
    # plt.plot(freq_final, df_off['psd'].iloc[ind], 'y', alpha=0.5)
    #plt.show()
    # calib_data_dict['zrms']
    
    psd_data = pd.DataFrame({'Frequency': freq_raw, 
                             'final': psd_raw, 
                             'laser on': df_on['psd'],#[ind0:ind1],
                             'laser off': df_off['psd'],#[ind0:ind1]
                            })
    psd_df_long = pd.melt(psd_data, id_vars=['Frequency'], value_vars=['laser on', 'laser off', 'final'],
                         var_name='name', value_name='PSD')
    
    zrms_i = (df_on['zrms']**2 - df_off['zrms']**2)**0.5 #CHECK THIS!
    
    #guess = [0, 76000, 2000, 100000]
    y_guess = 0 #psd_final.min()
    f_guess = freq_final[psd_final.argmax()]
    w_guess = 2*np.abs(freq_final[(np.abs(psd_final - psd_final.max()/2)).argmin()]-f_guess)
    A_guess = np.pi*w_guess*psd_final.max()/2
    guess = [y_guess, f_guess, w_guess, A_guess] #y0,f0,w,A
    # print(guess)
    #fit
    popt, pcov = curve_fit(ftf.lorentzian, freq_final,psd_final,
                        p0=guess, bounds=(0,np.inf), method='trf', ftol=1e-14, xtol=1e-14, gtol=1e-14)
    #print(np.linalg.cond(pcov))
    params = ['offset','resonance freq', 'fwhm', 'area']
    fit_dict = dict(zip(params, popt))
    fit_dict['Q factor'] = fit_dict['resonance freq']/fit_dict['fwhm']
    
    #plot fit
    f_min, f_max = freq_raw.min(), freq_raw.max()
    f_ext = 0.0*(f_max-f_min)
    fit_n = 8000
    freq_fit_range = np.linspace(f_min-f_ext, f_max+f_ext, fit_n)
    # plt.plot(freq_fit_range,ftf.lorentzian(freq_fit_range, *popt), 'k--')
    # psd_df_fit = pd.DataFrame({'Frequency': freq_fit_range, 'name': ['fit']*fit_n, 
    #                           'value': ftf.lorentzian(freq_fit_range, *popt)})
    # psd_df_all = pd.concat([psd_df_long, psd_df_fit])
    
    # fig = plotly_lineplot(data=psd_df_long, x="Frequency", y="PSD", color="name",
    #                      color_discrete_sequence=['blue', 'skyblue', 'red'])#, 'yellow'])
    # plotly_dashedlines(plot_type='line',fig=fig, x=freq_fit_range, 
    #                    y=ftf.lorentzian(freq_fit_range, *popt), line_width=2, name='fit')
    # fig.update_layout(legend_title=None)

    fig = seaborn_lineplot(data=psd_df_long, x="Frequency", y="PSD", color="name", 
                           palette=['blue', 'skyblue', 'red'], line_width=3)
    matplotlib_dashedlines(plot_type='line',fig=fig, x=freq_fit_range, 
                           y=ftf.lorentzian(freq_fit_range, *popt), line_width=2, name='fit')
    # fig.data[-1].line.dash = 'dash'
    # plt.close()
    fig_html = fig2html(fig, plot_type='matplotlib')
    # print(fit_dict)

    # k_lever = kcant_sader_normal(width=cant_width, length=cant_length, Q_factor=fit_dict['Q factor'], 
    #                              freq_res=fit_dict['resonance freq'])
    sader_gci_output = kcant_sader_gci(lever_number=lever_number, Q_factor=fit_dict['Q factor'],
                                       freq_res=fit_dict['resonance freq'], userid=userid, password=password)
    k_lever = sader_gci_output['k_sader']
    # Q = fit_dict['Q factor'] #head_data['Quality factor (Q)']
    # k_lever = 2 # N/m
    # T = 300 #K
    kb = 1.380649e-23 #J/K
    fit_dict['V rms'] = np.sqrt(fit_dict['area']) #area under PSD is the total power, which is the square of rms value of signal
    fit_dict['zrms'] = zrms_i #true rms amplitude
    # corr_fac = 4/3 #Butt-Jaschke correction for thermal noise
    sens = np.sqrt(kb*T/k_lever)/fit_dict['V rms']/1e-9 #nm/V 
    # print(fit_dict)
    return sens, k_lever, fit_dict, fig_html

# Sader method using the Global Calibration Initiative API
def kcant_sader_gci(lever_number, Q_factor, freq_res, userid, password):
    sader_gci_output = SaderGCI_CalculateK(userid, password, lever_number,
                                           freq_res/1000, #in kHz
                                           Q_factor)
    return sader_gci_output

# Sader method for normal spring constant calibration of rectangular cantilevers
# width and length (of cantilever) in μm, freq_res (resonance frequency) in Hz
# dens (fluid density) in kg/m3, visc (fluid viscosity) in kg/m/s
# reference: https://github.com/GrandlLab/sader_calibration
def kcant_sader_normal(width, length, Q_factor, freq_res, dens = 1.18, visc = 1.86e-5):
    p = dens #fluid density (kg/m3)
    n = visc #fluid viscosity (kg/m/s)

    def Reynolds_num(f0,width): 
        result = (p * f0 * width**2) / (4 * n)
        return result
    
    def hydrodynamic_circ(f0,width,Re):
        # result = 1 + (4 * 1j * besselk(1, -1j*np.sqrt(1j*Re)))/\
        # (np.sqrt(1j*Re) * besselk(0, -1j*np.sqrt(1j*Re)))
        result = 1 + (4 * 1j * kv(1, -1j*np.sqrt(1j*Re)))/\
        (np.sqrt(1j*Re) * kv(0, -1j*np.sqrt(1j*Re)))
    
        return result

    def G_real(z):
        result = (0.91324 - 0.48274*z + 0.46842*z**2 - 0.12886*z**3\
                  + 0.044055*z**4 - 0.0035117*z**5 + 0.00069085*z**6) /\
                  (1 - 0.56964*z + 0.48690*z**2 - 0.13444*z**3\
                   + 0.045155*z**4 - 0.0035862*z**5 + 0.00069085*z**6)
        return result

    def G_imag(z):
        result = (-0.024134 - 0.029256*z + 0.016294*z**2 - 0.00010961*z**3\
                  + 0.000064577*z**4 - 0.00004451*z**5)/\
                  (1 - 0.59702*z + 0.55182*z**2 - 0.18357*z**3\
                   + 0.079156*z**4 - 0.014369*z**5 + 0.0028361*z**6)
        return result
    
    def hydrodynamic_corr(Re):
        logRe = np.log10(Re)
        result = G_real(logRe) + 1j*G_imag(logRe)
        return result

    def hydrodynamic_rect(f0,width):
        Re = Reynolds_num(f0,width)
        result = hydrodynamic_corr(Re) * hydrodynamic_circ(f0,width,Re)
    
        return result

    w = width * 1e-6
    l = length * 1e-6
    f0 = 2*np.pi*freq_res
    Q = Q_factor
    k_norm = 0.1906 * f0**2 * p * w**2 * l * np.imag(hydrodynamic_rect(f0,w)) * Q
    
    return k_norm
 