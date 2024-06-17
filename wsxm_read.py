import struct
import os
import re
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wsxm_analyze import convert_spectro2df, get_imgdata, SPECT_DICT
from plot_funcs import plotly_lineplot, plotly_heatmap, fig2html, imagedf_to_excel
import transform_funcs as tsf

DATA_TYPES = {'short':(2,'h'),'short-data':(2,'h'), 'unsignedshort':(2,'H'),
              'integer-data':(4,'i'), 'signedinteger':(4,'i'),
              'float-data':(4,'f'), 'double':(8,'d')}

WSXM_CHANNEL_DICT = {'top':'Topography', 'ch1': 'Normal force', 'ch2': 'Lateral force', 
                     'ch12': 'Excitation frequency', 'ch15': 'Amplitude', 'ch16': 'Phase',
                     'adh': 'Adhesion', 'sti': 'Stiffness'
                    }

def wsxm_get_common_files(filepath):
    # filepath = 'data/interdigThiols_tipSi3nN_b_0026.fb.ch1.gsi'
    path_dir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    # filename_com = os.path.basename(filepath).split('.')[0] #common file name
    match = re.search(r'\_\d{4}', filename) #regex to find 4 digit number in filename
    if match == None: #return same file for no matches #CHECK
        return [filepath] #print(filename)
    else:
        filename_com = filename[:match.start()+5]
    # print(filename_com)
    files = []
    for i in os.listdir(path_dir):
        path_i = os.path.join(path_dir,i)
        if os.path.isfile(path_i) and i.startswith(filename_com):
            files.append(path_i)    
    files.remove(filepath) #make sure filepath is the first item in the list
    files.insert(0, filepath)
    return files


#read WSxM header data
def wsxm_readheader(file, pos=0, inibyte=100):
    header_dict = {}
    title_list = []
    # Find header size
    file.seek(pos, 0)
    data = file.read(inibyte)
    for ln in data.splitlines():
        hd_lst = ln.decode('latin-1', errors='ignore').split(':')
        if len(hd_lst) == 2:
            if hd_lst[0] == 'Image header size':
                header_size = int(hd_lst[1])
                # print(header_size)
                break
    # read header data (NOTE THAT DUPLICATE HEADER NAMES (example P,I values) WILL BE SKIPPED!
    file.seek(pos, 0)
    data = file.read(header_size)#[:header_size]
    for ln in data.splitlines():
        hd_lst = ln.decode('latin-1', errors='ignore').split(':')
        if len(hd_lst) == 2:
            # header_name = hd_lst[0].strip()
            # if header_name in header_dict.keys():
            #     header_name = header_name + ' ' + header_dict['Header sections'][-1]
            header_name = f"{hd_lst[0].strip()} {title_list[-1]}".strip()
            header_dict[header_name] = hd_lst[1].strip()
        elif len(hd_lst) == 1 and hd_lst[0] != '': #collect section tiles in header file
            title_list.append(hd_lst[0])
    
    pos_new = pos + header_size #bytes read so far
    # print(header_dict)
    return header_dict, pos_new

#read WSxM binary image data
def wsxm_readimg(file, header_dict, pos):
    data_format = header_dict['Image Data Type [General Info]']
    chan_label = header_dict['Acquisition channel [General Info]']
    line_rate = float(header_dict['X-Frequency [Control]'].split(' ')[0])
    x_num = int(header_dict['Number of rows [General Info]'])
    y_num = int(header_dict['Number of columns [General Info]'])
    x_len = float(header_dict['X Amplitude [Control]'].split(' ')[0])
    y_len = float(header_dict['Y Amplitude [Control]'].split(' ')[0])
    z_len = float(header_dict['Z Amplitude [General Info]'].split(' ')[0])
    x_dir = header_dict['X scanning direction [General Info]']
    y_dir = header_dict['Y scanning direction [General Info]'] #CHECK Y DIRECTIONS
    #CHECK THIS FOR SECOND ARRAY! MAY NOT WORK FOR 3D Mode images!
    #THIS DOES NOT WORK. CHECK EVERYWHERE
    dsp_voltrange = float(header_dict['DSP voltage range [Miscellaneous]'].split(' ')[0])
    # chan_adc2v = 20/2**16
    # chan_fact = int(header_dict['Conversion Factor 00'].split(' ')[0])
    # chan_offs = 0#int(header_dict['Conversion Offset 00'].split(' ')[0])

    x_data = np.linspace(x_len, 0, x_num, endpoint=True) #if x_dir == 'Backward' else np.linspace(x_len, 0, x_num, endpoint=True)
    y_data = np.linspace(0, y_len, y_num, endpoint=True) #if y_dir == 'Down' else np.linspace(y_len, 0, y_num, endpoint=True)
    # xx_data, yy_data = np.meshgrid(x_data, y_data)
    
    #read binary image data
    point_length, type_code  = DATA_TYPES[data_format]
    # with open(filepath, 'rb') as file:
    file.seek(pos, 0)
    data_len = x_num*y_num*point_length
    bin_data = file.read(data_len)
    # print(data.read()[(x_num*y_num*point_length)+header_size:])
    ch_array = np.array(list(struct.iter_unpack(f'{type_code}', bin_data))).flatten()
    #dac to volt conversion
    if chan_label == 'Topography': #ignore for topo
        if z_len == 0: #for zero data
            z_calib = 1
            chan_fact = 1
            chan_offs = 0
        else:
            z_calib = z_len/(ch_array.max()-ch_array.min())
            chan_fact = 1
            chan_offs = 0
    else: #other channel data stored in volts
        z_calib = dsp_voltrange/(2**16)
        chan_fact = float(header_dict['Conversion Factor 00 [General Info]'].split(' ')[0])
        if chan_label == 'Excitation frequency': #for freq shift
            chan_offs = 0
        else:
            chan_offs = float(header_dict['Conversion Offset 00 [General Info]'].split(' ')[0])
    # z_calib2 = z_len/(ch_array.max()-ch_array.min())
    # print(z_calib, z_calib2, z_calib-z_calib2)
    
    #img data dictionary
    data_dict_chan = {'data': {'Z': chan_offs + (chan_fact*z_calib*ch_array.reshape(x_num, y_num)),
                               'X': x_data,
                               'Y': y_data},
                      'header': header_dict.copy()}
    
    pos += data_len #bytes read so far
    return data_dict_chan, pos
    
# Read WSxM channel image data
def wsxm_readchan(filepath, all_files=False, mute=False):
    if all_files == True: #find all channels and directions of this measurement
        filepath_all = wsxm_get_common_files(filepath)
    else:
        filepath_all = [filepath]
    data_dict = {}
    file_num = 1 #file number
    for path in filepath_all:
        path_ext = os.path.splitext(path)[1] #file extension
        if path_ext != 'gsi': #ignore *.gsi files sharing same name
            if all_files==True and mute == False:
                print(file_num, os.path.basename(path)) 
            file_num += 1
            file = open(f'{path}','rb')
            header_dict, pos = wsxm_readheader(file)
            chan_label = header_dict['Acquisition channel [General Info]']
            data_dict_chan, pos = wsxm_readimg(file, header_dict, pos)
            x_dir = header_dict['X scanning direction [General Info]']
            if chan_label in data_dict.keys():
                data_dict[chan_label][x_dir] = data_dict_chan
            else:
                data_dict[chan_label] = {}
                data_dict[chan_label][x_dir] = data_dict_chan
            file.close()
    if all_files == True:
        wsxm_calc_extrachans(data_dict, data_type='2D')
        return data_dict
    else: #only return the specifc data dictionary for single file if all files are not read
        return data_dict_chan

# read *.curves file with image and f-d curves
#TODO: read other spectro data (*.stp and *.cur) similarly and output it in the same format as data_dict below!
#TODO: apply Conversion Factor to final channel value. CHECK THIS EVERYWHERE!
def wsxm_readcurves(path):
    # if all_files == True: #find all channels and directions of this measurement
    #     filepath_all = wsxm_get_common_files(filepath)
    # else:
    #     filepath_all = [filepath]
    data_dict = {}
    # file_num = 1 #file number
    # for path in filepath_all:
    #     path_ext = os.path.splitext(path)[1] #file extension
    #     if path_ext == '.curves': # read *.curves spectroscopy files
    #         if all_files==True:
    #             print(file_num, os.path.basename(path)) 
    #         file_num += 1
    file = open(f'{path}','rb')
    header_dict_top, pos = wsxm_readheader(file)
    data_dict_chan, pos = wsxm_readimg(file, header_dict_top, pos) 
    
    data_format = header_dict_top['Image Data Type [General Info]']
    point_length, type_code  = DATA_TYPES[data_format]
    data_dict_curv = {}
    
    while True:
        # file.seek(pos, 0)
        header_dict, pos = wsxm_readheader(file, pos=pos)     
        line_pts = int(header_dict['Number of points [General Info]'])
        line_num = int(header_dict['Number of lines [General Info]'])
        y_label = header_dict['Y axis text [General Info]'].split('[')[0].strip()
        x_label = header_dict['X axis text [General Info]'].split('[')[0].strip()
        curv_ind = int(header_dict['Index of this Curve [Control]'])
        curv_num = int(header_dict['Number of Curves in this serie [Control]'])
        #CHECK THIS FOR SECOND ARRAY! MAY NOT WORK FOR 3D Mode!
        # chan_adc2v = 1#20/2**16 #adc to volt converter for 20V DSP, 16 bit resolution
        chan_fact = float(header_dict['Conversion Factor 00 [General Info]'].split(' ')[0])
        if y_label == 'Excitation frequency': # For frequency shift
            chan_offs = 0
        else:
            chan_offs = float(header_dict['Conversion Offset 00 [General Info]'].split(' ')[0])
        # chan_offs = float(header_dict['Conversion Offset 00 [General Info]'].split(' ')[0])
        
        aqpt_x, aqpt_y = tuple(map(float, header_dict['Acquisition point [Control]'].replace('nm','').
                                   replace('(','').replace(')','').split(',')))
        time_f = float(header_dict['Forward plot total time [Control]'].split(' ')[0])
        time_b = float(header_dict['Backward plot total time [Control]'].split(' ')[0])
        
        line_order = ['approach', 'retract']
        if header_dict['First Forward [Miscellaneous]'] == 'No': #CHECK THIS
            line_order = ['retract', 'approach']

        data_len = line_pts*line_num*2*point_length
        file.seek(pos, 0)
        bin_data = file.read(data_len)
        ch_array = np.array(list(struct.iter_unpack(f'{type_code}', bin_data))).flatten()
        x_data, y_data = np.split(ch_array[::2], 2), np.split(ch_array[1::2], 2)
        
        data_dict_curv[curv_ind] = {'header': header_dict_top.copy() | header_dict.copy(), 'data': {}} #merge header dictionaries
        for i, curv_dir in enumerate(line_order):                    
            data_dict_curv[curv_ind]['data'][curv_dir] = {'x': x_data[i].max()-x_data[i], #reverse x data
                                                          'y': chan_offs+(y_data[i]*chan_fact) #converted to proper units
                                                          }
                                                # 'segment':np.append(line_pts * [line_order[0]],line_pts * [line_order[1]])},
                                                  
        
        if curv_ind == curv_num:
            break
        else:
            pos += data_len #bytes read so far
            file.seek(pos, 0)

    data_dict[y_label] = {'image': data_dict_chan,
                          'curves': data_dict_curv
                          }
    file.close()
    
    return data_dict, y_label
    # if all_files == True:
    #     return data_dict
    # else: #only return the specifc data dictionary for single file if all files are not read
    #     return data_dict[y_label]['curves'][curv_ind]

# read *.cur WSxM file
def wsxm_readcur(path):
    # if all_files == True: #find all channels and directions of this measurement
    #     filepath_all = wsxm_get_common_files(filepath)
    # else:
    #     filepath_all = [filepath]
    data_dict = {}
    # file_num = 1 #file number
    # for path in filepath_all:
    #     path_ext = os.path.splitext(path)[1] #file extension
    #     if path_ext == '.cur': # read *.curves spectroscopy files
    #         if all_files==True:
    #             print(file_num, os.path.basename(path)) 
    #         file_num += 1
    file = open(f'{path}','rb')
    header_dict, pos = wsxm_readheader(file)
    # data_dict_chan, pos = wsxm_readimg(file, header_dict, pos) 
    
    # data_format = header_dict['Image Data Type']
    # point_length, type_code  = DATA_TYPES[data_format]
    # data_dict_curv = {}
    
    # while True:
    # file.seek(pos, 0)
    # header_dict, pos = wsxm_readheader(file, pos=pos)
    if 'Index of this Curve [Control]' in header_dict.keys(): #for spectroscopy curves
        line_pts = int(header_dict['Number of points [General Info]'])
        line_num = int(header_dict['Number of lines [General Info]'])
        y_label = header_dict['Y axis text [General Info]'].split('[')[0].strip()
        x_label = header_dict['X axis text [General Info]'].split('[')[0].strip()
        if header_dict['Index of this Curve [Control]'] == 'Average': #for average curves
            curv_ind = header_dict['Index of this Curve [Control]']
        else:
            curv_ind = int(header_dict['Index of this Curve [Control]'])
        curv_num = int(header_dict['Number of Curves in this serie [Control]'])
        #CHECK THIS FOR SECOND ARRAY! MAY NOT WORK FOR 3D Mode!
        # chan_adc2v = 1#20/2**16 #adc to volt converter for 20V DSP, 16 bit resolution
        chan_fact = float(header_dict['Conversion Factor 00 [General Info]'].split(' ')[0])
        if y_label == 'Excitation frequency': # For frequency shift
            chan_offs = 0
        else:
            chan_offs = float(header_dict['Conversion Offset 00 [General Info]'].split(' ')[0])
        # chan_offs = float(header_dict['Conversion Offset 00 [General Info]'].split(' ')[0])
        
        aqpt_x, aqpt_y = tuple(map(float, header_dict['Acquisition point [Control]'].replace('nm','').
                                   replace('(','').replace(')','').split(',')))
        time_f = float(header_dict['Forward plot total time [Control]'].split(' ')[0])
        time_b = float(header_dict['Backward plot total time [Control]'].split(' ')[0])
        
        line_order = ['approach', 'retract']
        if header_dict['First Forward [Miscellaneous]'] == 'No': #CHECK THIS
            line_order = ['retract', 'approach']
    else: #for other kinds of *.cur (e.g. tune data)
        line_pts = int(header_dict['Number of points [General Info]'])
        line_num = int(header_dict['Number of lines [General Info]'])
        y_label = header_dict['Y axis text [General Info]'].split('[')[0].strip()
        x_label = header_dict['X axis text [General Info]'].split('[')[0].strip()
        #set generic values for irrelevant parameters here
        curv_ind = 1
        curv_num = 1
        chan_fact = 1
        chan_offs = 0                
        aqpt_x, aqpt_y = 0, 0
        time_f = 0
        time_b = 0                
        line_order = [f'{y_label}_1', f'{y_label}_2']

    # data_len = line_pts*line_num*2*point_length
    file.seek(pos, 0)
    data = file.read()
    data_list = []
    for ln in data.splitlines():
        ln_array = ln.decode('latin-1', errors='ignore').strip().split(' ')
        # print(ln_array)
        data_list.append(list(map(float,ln_array)))
    data_mat = np.array(data_list) #data matrix   
    # print(data_mat)
    # ch_array = np.array(list(struct.iter_unpack(f'{type_code}', bin_data))).flatten()
    # x_data, y_data = np.split(ch_array[::2], 2), np.split(ch_array[1::2], 2)
    if y_label not in data_dict.keys():
        data_dict[y_label] = {'curves':{}, 'image':{}}
    data_dict[y_label]['curves'][curv_ind] = {'header': header_dict.copy(), 'data': {}}
    if 'Index of this Curve [Control]' in header_dict.keys(): #TODO: make "reverse data" as a function for transformation! Then eliminate if-else
        for i, curv_dir in enumerate(line_order):
            data_dict[y_label]['curves'][curv_ind]['data'][curv_dir] = {'x': data_mat[:,2*i].max()-data_mat[:,2*i], #reverse x data
                                                                        'y': chan_offs+(data_mat[:,2*i+1]*chan_fact) #converted to units
                                                                        }
    else: 
        for i, curv_dir in enumerate(line_order):
            data_dict[y_label]['curves'][curv_ind]['data'][curv_dir] = {'x': data_mat[:,2*i], #original x data
                                                                        'y': chan_offs+(data_mat[:,2*i+1]*chan_fact) #converted to units
                                                                        }

    file.close()
    
    return data_dict, y_label
    
    # if all_files == True:
    #     return data_dict
    # else: #only return the specifc data dictionary for single file if all files are not read
    #     return data_dict[y_label]['curves'][curv_ind]


#read *.stp spectroscopy curves. Use data_dict to update data of both approach and retract into the data dictionary
def wsxm_readstp(path, data_dict={}):
    # if all_files == True: #find all channels and directions of this measurement
    #     filepath_all = wsxm_get_common_files(filepath)
    # else:
    #     filepath_all = [filepath]
    # data_dict = {}
    # file_num = 1 #file number
    # for path in filepath_all:
    #     path_ext = os.path.splitext(path)[1] #file extension
    #     if path_ext == '.stp': # read *.stp spectroscopy files
    #         if all_files==True:
    #             print(file_num, os.path.basename(path)) 
    #         file_num += 1
    file = open(f'{path}','rb')
    filename = os.path.basename(path)
    header_dict, pos = wsxm_readheader(file)
    data_format = header_dict['Image Data Type [General Info]']
    chan_label = filename.split('_')[-1].split('.')[0] #header_dict['Acquisition channel']
    # line_rate = float(header_dict['X-Frequency'].split(' ')[0])
    x_num = int(header_dict['Number of rows [General Info]'])
    y_num = int(header_dict['Number of columns [General Info]'])
    x_len = float(header_dict['X Amplitude [Control]'].split(' ')[0])
    y_len = float(header_dict['Y Amplitude [Control]'].split(' ')[0])
    z_len = float(header_dict['Z Amplitude [General Info]'].split(' ')[0])
    x_dir = header_dict['X scanning direction [General Info]']
    y_dir = header_dict['Y scanning direction [General Info]'] #CHECK Y DIRECTIONS
    z_dir = SPECT_DICT[filename.split('.')[-2]]
    dsp_voltrange = float(header_dict['DSP voltage range [Miscellaneous]'].split(' ')[0])
    # print(z_dir,filename)
    # chan_fact = float(header_dict['Conversion Factor 00 [General Info]'].split(' ')[0])
    # if chan_label == 'Excitation frequency': # For frequency shift
    #     chan_offs = 0
    # else:
    #     chan_offs = float(header_dict['Conversion Offset 00 [General Info]'].split(' ')[0])

    z_data = np.linspace(0, x_len, y_num, endpoint=True) #CHECK THIS
    # print(filename,x_dir,y_dir,z_dir)
    #read binary image data
    point_length, type_code  = DATA_TYPES[data_format]
    # with open(filepath, 'rb') as file:
    file.seek(pos, 0)
    data_len = x_num*y_num*point_length
    bin_data = file.read(data_len)
    # print(data.read()[(x_num*y_num*point_length)+header_size:])
    ch_array = np.array(list(struct.iter_unpack(f'{type_code}', bin_data))).flatten() 
    ch_mat = ch_array.reshape(x_num,y_num)
    if z_len == 0: #for zero data
        z_calib = 1
    else:
        # z_calib = chan_fact*dsp_voltrange/(2**16)
        z_calib = z_len/(ch_array.max()-ch_array.min())
    
    #create separate curve data for each line (consistent with '1D' data format)
    for i in range(x_num): 
        curv_ind = i + 1        
        #data dictionary initialised in a consistant format (also check wsxm_readcurves())
        if chan_label not in data_dict.keys():
            data_dict[chan_label] = {'curves': {}, 'image':{}}
        if curv_ind not in data_dict[chan_label]['curves'].keys():
            data_dict[chan_label]['curves'][curv_ind] = {'data': {},'header': header_dict.copy()}
            #insert curve number info into header
            data_dict[chan_label]['curves'][curv_ind]['header']['Index of this Curve [Control]'] = str(curv_ind) 
            data_dict[chan_label]['curves'][curv_ind]['header']['Number of Curves in this serie [Control]'] = str(x_num)
        if z_dir not in data_dict[chan_label]['curves'][curv_ind]['data'].keys():
            data_dict[chan_label]['curves'][curv_ind]['data'][z_dir] = {}
        data_dict[chan_label]['curves'][curv_ind]['data'][z_dir]['x'] = z_data.max()-z_data #reverse x data
        data_dict[chan_label]['curves'][curv_ind]['data'][z_dir]['y'] = (z_calib*ch_mat[:][i]) #chan_offs+(ch_mat[:][i]*chan_fact)
        if x_dir == 'Forward':
            data_dict[chan_label]['curves'][curv_ind]['data'][z_dir]['y'] = np.flip((z_calib*ch_mat[:][i]))

    file.close()
    return data_dict, chan_label
    
    # if all_files == True:
    #     return data_dict
    # else:  #only return the specifc data dictionary for single file if all files are not read
    #     return data_dict[chan_label]['curves'][curv_ind]
    
# Read WSxM 1D spectroscopy data and curves for all available channels
def wsxm_readspectra(filepath, all_files=False, mute=False):
    # if all_files == True: #find all channels and directions of this measurement
    filepath_all = wsxm_get_common_files(filepath)
    path_ext_f = os.path.splitext(filepath)[1]
    data_dict = {}
    data_dict_stp = {}
    file_num = 1 #file number
    for path in filepath_all:
        # print('yo', path, data_dict.keys())
        path_ext = os.path.splitext(path)[1] #file extension
        if all_files==True and mute == False:
            print(file_num, os.path.basename(path)) 
        file_num += 1
        if all_files == False and path_ext == path_ext_f: #collect all curve for the same file type (eg. approach/retract)
            if path_ext == '.curves': # read *.curves spectroscopy files
                temp_dict, chan_label = wsxm_readcurves(path)
                data_dict[chan_label] = temp_dict[chan_label].copy()
            elif path_ext == '.stp': # read *.stp spectroscopy files
                temp_dict, chan_label = wsxm_readstp(path, data_dict)#data_dict)
                # if chan_label not in data_dict.keys(): #ignore data if *.curves already found
                # data_dict[chan_label] = temp_dict[chan_label]
            elif path_ext == '.cur': # read *.cur spectroscopy files
                temp_dict, chan_label = wsxm_readcur(path)
                # if chan_label not in data_dict.keys(): #ignore data if *.curves already found
                data_dict[chan_label] = temp_dict[chan_label].copy()
            if path == filepath:
                chan_label_f = chan_label[:]
                curv_ind_f = list(temp_dict[chan_label_f]['curves'].keys())[0] #temp_dict[chan_label]['header']['Index of this Curve [Control]']
        elif all_files == True:
            if path_ext == '.curves': # read *.curves spectroscopy files
                temp_dict, chan_label = wsxm_readcurves(path)
                if chan_label not in data_dict.keys():
                    data_dict[chan_label] = temp_dict[chan_label].copy()
                else:
                    for curv_ind_i in temp_dict[chan_label]['curves'].keys(): #replace with *.curves data even if it already exists (more robust)
                        data_dict[chan_label]['curves'][curv_ind_i] = temp_dict[chan_label]['curves'][curv_ind_i].copy()
            elif path_ext == '.stp': # read *.stp spectroscopy files
                temp_dict, chan_label = wsxm_readstp(path, data_dict_stp)
                if chan_label not in data_dict.keys(): #ignore data if *.curves already found
                    data_dict[chan_label] = temp_dict[chan_label].copy()
                else:
                    for curv_ind_i in temp_dict[chan_label]['curves'].keys():
                        if curv_ind_i not in data_dict[chan_label]['curves'].keys():
                            data_dict[chan_label]['curves'][curv_ind_i] = temp_dict[chan_label]['curves'][curv_ind_i].copy()
            elif path_ext == '.cur': # read *.cur spectroscopy files
                temp_dict, chan_label = wsxm_readcur(path)
                if chan_label not in data_dict.keys(): #ignore data if *.curves already found
                    data_dict[chan_label] = temp_dict[chan_label].copy()
                else:
                    for curv_ind_i in temp_dict[chan_label]['curves'].keys():
                        if curv_ind_i not in data_dict[chan_label]['curves'].keys():
                            data_dict[chan_label]['curves'][curv_ind_i] = temp_dict[chan_label]['curves'][curv_ind_i].copy()
        # data_dict[chan_label] = temp_dict[chan_label]
        # print('hey', path, filepath)
        
    
    if all_files == True:
        wsxm_calc_extrachans(data_dict, data_type='1D')
        return data_dict
    else:  #only return the specifc data dictionary for single file if all files are not read
        return data_dict[chan_label_f]['curves'][curv_ind_f]
            
# Read WSxM Force volume data
def wsxm_readforcevol(filepath, all_files=False, topo_only=False):
    if all_files == True: #find all channels and directions of this measurement
        filepath_all = wsxm_get_common_files(filepath)
    else:
        filepath_all = [filepath]
    data_dict = {}
    file_num = 1 #file number
    for path in filepath_all:
        path_ext = os.path.splitext(path)[1] #file extension
        # if path_ext == '.top': #topgraphy data
        #     data_dict['Topography'] = wsxm_readchan(path)
        if path_ext == '.gsi': #force volume data from *.gsi files
            if all_files==True:
                print(file_num, os.path.basename(path)) 
            file_num += 1
            file = open(f'{path}','rb')
            header_dict, pos = wsxm_readheader(file)
            
            data_format = header_dict['Image Data Type [General Info]']
            chan_label = header_dict['Acquisition channel [General Info]']
            spec_dir = header_dict['Spectroscopy type [General Info]']
            x_dir = spec_dir.split(' ')[1]
            y_dir = header_dict['Y scanning direction [General Info]'] #CHECK Y DIRECTIONS
            # z_dir = SPECT_DICT[spec_dir.split(' ')[3]]
            line_rate = float(header_dict['X-Frequency [Control]'].split(' ')[0])
            x_num = int(header_dict['Number of rows [General Info]'])
            y_num = int(header_dict['Number of columns [General Info]'])
            chan_num = int(header_dict['Number of points per ramp [General Info]'])
            x_len = float(header_dict['X Amplitude [Control]'].split(' ')[0])
            y_len = float(header_dict['Y Amplitude [Control]'].split(' ')[0])
            z_len = float(header_dict['Z Amplitude [General Info]'].split(' ')[0])
            chan_adc2v = float(header_dict['ADC to V conversion factor [General Info]'].split(' ')[0])
            chan_fact = float(header_dict['Conversion factor 0 for input channel [General Info]'].split(' ')[0])
            if chan_label == 'Excitation frequency': # For frequency shift
                chan_offs = 0
            else:
                chan_offs = float(header_dict['Conversion offset 0 for input channel [General Info]'].split(' ')[0])
            # chan_offs = float(header_dict['Conversion offset 0 for input channel [General Info]'].split(' ')[0])
                    
            x_data = np.linspace(x_len, 0, x_num, endpoint=True) #if x_dir == 'Backward' else np.linspace(x_len, 0, x_num, endpoint=True)
            y_data = np.linspace(0, y_len, y_num, endpoint=True) #if y_dir == 'Down' else np.linspace(y_len, 0, y_num, endpoint=True)
            # xx_data, yy_data = np.meshgrid(x_data, y_data)
        
            z_data = np.empty(0)
            for i in range(chan_num):
                z_data = np.append(z_data, float(header_dict[f'Image {i:03} [Spectroscopy images ramp value list]'].split(' ')[0]))
            # if z_dir == 'retract':
            z_data = np.flip(z_data) #reverse z data order to make zero as point of contact
            
            #read binary image data
            point_length, type_code  = DATA_TYPES[data_format]
            # with open(filepath, 'rb') as file:
            file.seek(pos, 0)
            data_len = x_num*y_num*point_length
            # pos += data_len #skip first topo image
            #read first topography data
            bin_data = file.read(data_len)
            topo_array = np.array(list(struct.iter_unpack(f'{type_code}', bin_data))).flatten()
            if z_len == 0: #for zero data
                topo_calib = 1
            else:
                topo_calib = z_len/(topo_array.max()-topo_array.min())
            #topo data dictionary
            data_dict_topo = {'data': {'Z': topo_calib*topo_array.reshape(x_num, y_num),
                                       'X': x_data,
                                       'Y': y_data
                                       },
                              'header': header_dict}
            topo_label = 'Topography'
            
            if topo_only == True and all_files == False: #return only topo data dictionary
                file.close()
                return data_dict_topo
                
            if topo_label not in data_dict.keys():
                data_dict[topo_label] = {}
            data_dict[topo_label][spec_dir] = data_dict_topo
            
            if topo_only == False: #skip channel read if topo_only=True
                pos += data_len
                ch_array = np.empty(0) #initialize channel data array
                for i in range(1, chan_num+1):
                    file.seek(pos, 0)
                    bin_data = file.read(data_len)
                    # print(data.read()[(x_num*y_num*point_length)+header_size:])
                    ch_array_temp = np.array(list(struct.iter_unpack(f'{type_code}', bin_data))).flatten()
                    # print(ch_array_temp.min(), ch_array_temp.max())
                    # if i == 0:
                    #     z_calib = z_len/(ch_array_temp.max()-ch_array_temp.min())
                    # else:
                    ch_array = np.append(ch_array, chan_offs+(ch_array_temp*chan_adc2v*chan_fact))
                    pos += data_len #next image
                # print(z_calib, chan_adc2v, z_len)
                
                #img data dictionary
                data_dict_chan = {'data': {'ZZ': ch_array.reshape(x_num,y_num,chan_num),
                                           'X': x_data,
                                           'Y': y_data,
                                           'Z': z_data
                                          },
                                  'header': header_dict}
                if chan_label not in data_dict.keys():
                    data_dict[chan_label] = {}
                data_dict[chan_label][spec_dir] = data_dict_chan
            file.close()
        
        # pos += data_len #bytes read so far  
    wsxm_calc_extrachans(data_dict, data_type='3D')
    return data_dict

# add additional channels to data_dict
def wsxm_calc_extrachans(data_dict, data_type):
    channels = data_dict.keys()
    #Include into data_dict true amplitude and true phase from the "amplitude" and "phase" channels, 
    #which are in-fact the quadrature and in-phase outputs, respectively,of the lock-in amplifier
    if all(chan in channels for chan in ['Amplitude', 'Phase']) == True:
        amp_data = data_dict['Amplitude']
        phase_data = data_dict['Phase']
        data_dict['True Amplitude'] = {}
        data_dict['True Phase'] = {}
        if data_type == '1D':
            data_dict['True Amplitude']['curves'] = {}
            data_dict['True Phase']['curves'] = {}
            for amp_i, phase_i in zip(amp_data['curves'].items(), phase_data['curves'].items()):
                data_dict['True Amplitude']['curves'][amp_i[0]] = {'data':{'approach':{'x':amp_i[1]['data']['approach']['x'],
                                                                                       'y':tsf.hypotenuse(amp_i[1]['data']['approach']['y'],
                                                                                                          phase_i[1]['data']['approach']['y'])
                                                                                      },
                                                                           'retract':{'x':amp_i[1]['data']['retract']['x'],
                                                                                      'y':tsf.hypotenuse(amp_i[1]['data']['retract']['y'],
                                                                                                         phase_i[1]['data']['retract']['y'])
                                                                                     }
                                                                          },
                                                                   'header':amp_i[1]['header']
                                                                  }
                
                data_dict['True Phase']['curves'][phase_i[0]] = {'data':{'approach':{'x':phase_i[1]['data']['approach']['x'],
                                                                                   'y':np.arctan2(amp_i[1]['data']['approach']['y'],
                                                                                                  phase_i[1]['data']['approach']['y']*180/np.pi)
                                                                                      },
                                                                       'retract':{'x':phase_i[1]['data']['retract']['x'],
                                                                                  'y':np.arctan2(amp_i[1]['data']['retract']['y'],
                                                                                                 phase_i[1]['data']['retract']['y']*180/np.pi)
                                                                                     }
                                                                          },
                                                               'header':phase_i[1]['header']
                                                              }
        elif data_type == '2D':
            for amp_i, phase_i in zip(amp_data.items(), phase_data.items()):
                img_dir = amp_i[0]
                data_dict['True Amplitude'][img_dir] = {'data': {'X':amp_i[1]['data']['X'],
                                                                 'Y':amp_i[1]['data']['Y'],
                                                                 'Z':tsf.hypotenuse(amp_i[1]['data']['Z'],
                                                                                    phase_i[1]['data']['Z'])},
                                                        'header':amp_i[1]['header']
                                                       }

                data_dict['True Phase'][img_dir] = {'data': {'X':phase_i[1]['data']['X'],
                                                             'Y':phase_i[1]['data']['Y'],
                                                             'Z':np.arctan2(amp_i[1]['data']['Z'],
                                                                            phase_i[1]['data']['Z'])*180/np.pi},
                                                    'header':phase_i[1]['header']
                                                   }

        elif data_type == '3D':
            for amp_i, phase_i in zip(amp_data.items(), phase_data.items()):
                img_dir = amp_i[0]
                data_dict['True Amplitude'][img_dir] = {'data': {'X':amp_i[1]['data']['X'],
                                                                 'Y':amp_i[1]['data']['Y'],
                                                                 'Z':amp_i[1]['data']['Z'],
                                                                 'ZZ':tsf.hypotenuse(amp_i[1]['data']['ZZ'],
                                                                                     phase_i[1]['data']['ZZ'])},
                                                        'header':amp_i[1]['header']
                                                       }

                data_dict['True Phase'][img_dir] = {'data': {'X':phase_i[1]['data']['X'],
                                                             'Y':phase_i[1]['data']['Y'],
                                                             'Z':phase_i[1]['data']['Z'],
                                                             'ZZ':np.arctan2(amp_i[1]['data']['ZZ'],
                                                                             phase_i[1]['data']['ZZ'])*180/np.pi},
                                                    'header':phase_i[1]['header']
                                                   }
    else:
        chan_missing = ['Amplitude', 'Phase'][list(chan in channels for chan in ['Amplitude', 'Phase']).index(False)]
        print(f'True Amplitude/Phase channels not created due to missing channel: {chan_missing}')
     
    # add normal deflection channel, to be calibrated in length units
    if 'Normal force' in channels:
        data_dict['Normal deflection'] = dict(data_dict['Normal force']) #deep copy of normal force channel
    else:
        print(f'Normal deflection channel not created due to missing channel: Normal force')

#reads all wsxm data files in a folder, collects them into a table with thumbnails and file metadata information for browsing.
#saved the table as a binary and excel file in the folder. The binary file can be later loaded directly to avoid reading all the files again.
#"refresh" parameter can be used to search the directory again for file and replace existing pickle/excel file list
# def wsxm_collect_files(folderpath, refresh=False):
#     # folderpath = 'data/'
#     # folderpath = filedialog.askdirectory() #use folder picker dialogbox
#     picklepath = f"{folderpath}/filelist_{os.path.basename(folderpath)}.pkl" #pickled binary file
#     if os.path.exists(picklepath) and refresh==False:
#         file_df = pd.read_pickle(picklepath) #choose "datalist.pkl" file (faster)
#     else:
#         file_dict = {'plot': [], 'file':[], 'name': [], 'channel': [], 'type': [], #'mode': [], 
#                      'feedback': [], 'size':[], 'resolution':[], 'time':[]}
#         for filename_i in os.listdir(folderpath):
#             path_i = os.path.join(folderpath,filename_i)
#             if os.path.isfile(path_i):
#                 match_i = re.search(r'\_\d{4}', filename_i) #regex to find 4 digit number in filename
#                 time_i = datetime.datetime.fromtimestamp(os.path.getmtime(path_i)) #time of file modified (from file metadata)
#                 path_ext_i = os.path.splitext(path_i)[1] #file extension
#                 if path_ext_i in ['.pkl','.xlsx','.txt']: #ignore pickle and excel and other files
#                     continue
#                 if match_i != None:
#                     # print(datetime.datetime.now().strftime("%H:%M:%S"), filename_i)
#                     filename_com_i = filename_i[:match_i.start()+5]
#                     if path_ext_i == '.gsi':
#                         data_type_i = '3D'
#                         channel_i = 'Topography' #only check topo image for force volume data
#                         feedback_i = ''
                        
#                         data_dict_chan_i = wsxm_readforcevol(path_i, all_files=False, topo_only=True)
#                         header_i = data_dict_chan_i['header']
#                         # print(header_i)
#                         z_pts_i = int(header_i['Number of points per ramp'])
#                         z_extrema_i = [float(header_i[f'Image {z_pts_i-1:03}'].split(' ')[0]),
#                                        float(header_i['Image 000'].split(' ')[0])]
#                         res_i = header_i['Number of columns'] + 'x' + header_i['Number of columns'] + 'x' + header_i['Number of points per ramp']
#                         size_i = header_i['X Amplitude'] + ' x ' + header_i['Y Amplitude'] + ' x ' + f'{int(max(z_extrema_i))}' + ' ' + header_i['Image 000'].split(' ')[1]
#                         xx_i, yy_i, zz_i = get_imgdata(data_dict_chan_i)
#                         plt.pcolormesh(xx_i, yy_i, zz_i, cmap='afmhot')
#                         plt.axis('off')
#                         fig_i = fig2html(plt.gcf())
#                         plt.close()
#                     elif path_ext_i in ['.curve']: #TODO: *.curve also combine to below condition!
#                         data_type_i = '1D'
#                         channel_i = filename_i[match_i.start()+6:].split('.')[0].split('_')[0]
#                         fig_i = ''
#                         res_i = ''
#                         size_i = ''
#                         feedback_i = ''
#                     elif path_ext_i in ['.curves', '.stp', '.cur']:
#                         data_type_i = '1D'
#                         channel_i = filename_i[match_i.start()+6:].split('.')[0].split('_')[0]
#                         feedback_i = ''
#                         data_dict_chan_i = wsxm_readspectra(path_i, all_files=False)
#                         spec_dir_i = list(data_dict_chan_i['data'].keys())
#                         header_i = data_dict_chan_i['header']
#                         if path_ext_i == '.stp':                            
#                             res_i = header_i['Number of columns']
#                             size_i = header_i['X Amplitude']
#                         else: #for *.curves and *.cur
#                             res_i = header_i['Number of points']
#                             size_i = str(data_dict_chan_i['data'][spec_dir_i[0]]['x'].max())  + ' ' + header_i['X axis unit']
#                         # if path_ext_i == '.curves':
#                         #     data_dict_chan_i = wsxm_readcurves(path_i, all_files=False)
#                         #     header_i = data_dict_chan_i['header']
#                         #     res_i = header_i['Number of points']
#                         #     spec_dir_i = list(data_dict_chan_i['data'].keys())
#                         #     size_i = str(data_dict_chan_i['data'][spec_dir_i[0]]['x'].max())  + ' ' + header_i['X axis unit']
#                         # elif path_ext_i == '.cur':
#                         #     data_dict_chan_i = wsxm_readcur(path_i, all_files=False)
#                         #     header_i = data_dict_chan_i['header']
#                         #     res_i = header_i['Number of points']
#                         #     spec_dir_i = list(data_dict_chan_i['data'].keys())
#                         #     size_i = str(data_dict_chan_i['data'][spec_dir_i[0]]['x'].max())  + ' ' + header_i['X axis unit']
#                         # elif path_ext_i == '.stp':
#                         #     data_dict_chan_i = wsxm_readstp(path_i, all_files=False)
#                         #     header_i = data_dict_chan_i['header']
#                         #     res_i = header_i['Number of columns']
#                         #     spec_dir_i = list(data_dict_chan_i['data'].keys())
#                         #     size_i = header_i['X Amplitude']
#                         spectrodf_i = convert_spectro2df(data_dict_chan_i['data'])
#                         sns.lineplot(data=spectrodf_i, x="x", y="y", hue="segment")
#                         fig_i = fig2html(plt.gcf())
#                         plt.close()
#                     else:
#                         data_type_i = '2D'
#                         channel_i = WSXM_CHANNEL_DICT[path_ext_i[1:]]
#                         file_tags_i = filename_i[match_i.start()+6:].split('.')
                        
#                         data_dict_chan_i = wsxm_readchan(path_i, all_files=False)
#                         header_i = data_dict_chan_i['header']
#                         res_i = header_i['Number of rows'] + 'x' + header_i['Number of columns']
#                         size_i = header_i['X Amplitude'] + ' x ' + header_i['Y Amplitude']
#                         feedback_i = header_i['Input channel']
#                         xx_i, yy_i, zz_i = get_imgdata(data_dict_chan_i)
#                         plt.pcolormesh(xx_i, yy_i, zz_i, cmap='afmhot')
#                         plt.axis('off')
#                         fig_i = fig2html(plt.gcf())
#                         plt.close()
#                 else: #if no match for 4 digit counter found in file name
#                     if path_ext_i == '.cur': #read other *.cur file e.g. tuning
#                         filename_com_i = filename_i[:-4]
#                         data_type_i = '1D'
#                         channel_i = 'Other'
#                         data_dict_chan_i = wsxm_readspectra(path_i, all_files=False)
#                         header_i = data_dict_chan_i['header']
#                         res_i = header_i['Number of points']
#                         spec_dir_i = list(data_dict_chan_i['data'].keys())
#                         size_i = str(data_dict_chan_i['data'][spec_dir_i[0]]['x'].max())  + ' ' + header_i['X axis unit']
#                         feedback_i = ''
#                         spectrodf_i = convert_spectro2df(data_dict_chan_i['data'])
#                         sns.lineplot(data=spectrodf_i, x="x", y="y", hue="segment")
#                         fig_i = fig2html(plt.gcf())
#                         plt.close()
    
#                 file_dict['file'].append(filename_com_i)
#                 file_dict['name'].append(filename_i)
#                 file_dict['channel'].append(channel_i)
#                 file_dict['type'].append(data_type_i)
#                 file_dict['size'].append(size_i)
#                 file_dict['resolution'].append(res_i)
#                 file_dict['feedback'].append(feedback_i)
#                 file_dict['plot'].append(fig_i)
#                 file_dict['time'].append(time_i)
        
#         file_df = pd.DataFrame.from_dict(file_dict)

#         #save "pickled" binary data of file list for later use
#         file_df.to_pickle(f"{folderpath}/filelist_{os.path.basename(folderpath)}.pkl")
#         #save excel file for manual check
#         file_df.drop(columns=['plot']).to_excel(f"{folderpath}/filelist_{os.path.basename(folderpath)}.xlsx")
    
#     return file_df


#reads all wsxm data files in a folder, collects them into a table with thumbnails and file metadata information for browsing.
#saved the table as a binary and excel file in the folder. The binary file can be later loaded directly to avoid reading all the files again.
#"refresh" parameter can be used to search the directory again for file and replace existing pickle/excel file list
def wsxm_collect_files(folderpath, refresh=False, flatten_chan=[]):
    # folderpath = 'data/'
    # folderpath = filedialog.askdirectory() #use folder picker dialogbox
    picklepath = f"{folderpath}/filelist_{os.path.basename(folderpath)}.pkl" #pickled binary file
    if os.path.exists(picklepath) and refresh==False:
        file_df = pd.read_pickle(picklepath) #choose "datalist.pkl" file (faster)
    else:
        file_dict = {'plot': [], 'file':[], 'name': [], 'channel': [], 'type': [], #'feedback': [], #'mode': [], 
                     'size':[], 'resolution':[], 'max':[], 'min':[], 'avg':[], 'time':[], 
                     'extension':[], 'header': [], 'header names':[]}
        for filename_i in os.listdir(folderpath):
            path_i = os.path.join(folderpath,filename_i)
            if os.path.isfile(path_i):
                match_i = re.search(r'\_\d{4}', filename_i) #regex to find 4 digit number in filename
                time_i = datetime.datetime.fromtimestamp(os.path.getmtime(path_i)) #time of file modified (from file metadata)
                path_ext_i = os.path.splitext(path_i)[1] #file extension
                if path_ext_i in ['.pkl','.xlsx','.txt']: #ignore pickle and excel and other files
                    continue
                if match_i != None:
                    # print(datetime.datetime.now().strftime("%H:%M:%S"), filename_i)
                    filename_com_i = filename_i[:match_i.start()+5]
                    if path_ext_i == '.gsi':
                        data_type_i = '3D'
                        channel_i = 'Topography' #only check topo image for force volume data
                        # feedback_i = ''
                        
                        data_dict_chan_i = wsxm_readforcevol(path_i, all_files=False, topo_only=True)
                        header_i = data_dict_chan_i['header']
                        # print(header_i)
                        z_pts_i = int(header_i['Number of points per ramp [General Info]'])
                        z_extrema_i = [float(header_i[f'Image {z_pts_i-1:03} [Spectroscopy images ramp value list]'].split(' ')[0]),
                                       float(header_i['Image 000 [Spectroscopy images ramp value list]'].split(' ')[0])]
                        res_i = header_i['Number of rows [General Info]'] + 'x' + header_i['Number of columns [General Info]'] + 'x' + header_i['Number of points per ramp [General Info]']
                        size_i = header_i['X Amplitude [Control]'] + ' x ' + header_i['Y Amplitude [Control]'] + ' x ' + f'{int(max(z_extrema_i))}' + ' ' + header_i['Image 000 [Spectroscopy images ramp value list]'].split(' ')[1]
                        # xx_i, yy_i, zz_i = get_imgdata(data_dict_chan_i)
                        # plt.pcolormesh(xx_i, yy_i, zz_i, cmap='afmhot')
                        # plt.axis('off')
                        # fig_i = fig2html(plt.gcf(), plot_type='matplotlib')
                        # plt.close()
                        z_max_i = data_dict_chan_i['data']['Z'].max()
                        z_min_i = data_dict_chan_i['data']['Z'].min()
                        z_avg_i = data_dict_chan_i['data']['Z'].mean()
                        if flatten_chan == 'all' or channel_i in flatten_chan: #channel_i == 'Topography': #only flatten topography images
                            z_data_i = tsf.flatten_line(data_dict_chan_i['data'], order=1)
                        else:
                            z_data_i = data_dict_chan_i['data']['Z']
                        # z_data_i = tsf.flatten_line(data_dict_chan_i['data'], order=1) #flatten topography
                        fig_i = fig2html(plotly_heatmap(x=data_dict_chan_i['data']['X'],
                                                        y=data_dict_chan_i['data']['Y'],
                                                        z_mat=z_data_i, style='clean'), 
                                         plot_type='plotly')
                    elif path_ext_i in ['.curves', '.stp', '.cur']:
                        data_type_i = '1D'
                        channel_i = filename_i[match_i.start()+6:].split('.')[0].split('_')[0]
                        # feedback_i = ''
                        data_dict_chan_i = wsxm_readspectra(path_i, all_files=False)
                        spec_dir_i = list(data_dict_chan_i['data'].keys())
                        header_i = data_dict_chan_i['header']
                        if path_ext_i == '.stp':                            
                            res_i = header_i['Number of columns [General Info]']
                            size_i = header_i['X Amplitude [Control]']
                        else: #for *.curves and *.cur
                            res_i = header_i['Number of points [General Info]']
                            size_i = str(data_dict_chan_i['data'][spec_dir_i[0]]['x'].max())  + ' ' + header_i['X axis unit [General Info]']
                        spectrodf_i = convert_spectro2df(data_dict_chan_i['data'])
                        # sns.lineplot(data=spectrodf_i, x="x", y="y", hue="segment")
                        # fig_i = fig2html(plt.gcf())
                        z_max_i = spectrodf_i['y'].max()
                        z_min_i = spectrodf_i['y'].min()
                        z_avg_i = spectrodf_i['y'].mean()
                        fig_i = fig2html(plotly_lineplot(data=spectrodf_i, x="x", y="y", color="segment"), plot_type='plotly')
                        # plt.close()
                    else:
                        data_type_i = '2D'
                        channel_i = WSXM_CHANNEL_DICT[path_ext_i[1:]]
                        file_tags_i = filename_i[match_i.start()+6:].split('.')
                        
                        data_dict_chan_i = wsxm_readchan(path_i, all_files=False)
                        header_i = data_dict_chan_i['header']
                        res_i = header_i['Number of rows [General Info]'] + 'x' + header_i['Number of columns [General Info]']
                        size_i = header_i['X Amplitude [Control]'] + ' x ' + header_i['Y Amplitude [Control]']
                        # feedback_i = header_i['Input channel']
                        # xx_i, yy_i, zz_i = get_imgdata(data_dict_chan_i)
                        # plt.pcolormesh(xx_i, yy_i, zz_i, cmap='afmhot')
                        
                        # plt.axis('off')
                        z_max_i = data_dict_chan_i['data']['Z'].max()
                        z_min_i = data_dict_chan_i['data']['Z'].min()
                        z_avg_i = data_dict_chan_i['data']['Z'].mean()
                        if flatten_chan == 'all' or channel_i in flatten_chan: #channel_i == 'Topography': #only flatten topography images
                            z_data_i = tsf.flatten_line(data_dict_chan_i['data'], order=1)
                        else:
                            z_data_i = data_dict_chan_i['data']['Z']
                        fig_i = fig2html(plotly_heatmap(x=data_dict_chan_i['data']['X'],
                                                        y=data_dict_chan_i['data']['Y'],
                                                        z_mat=z_data_i, style='clean'), 
                                         plot_type='plotly')
                        
                        # plt.close()
                else: #if no match for 4 digit counter found in file name
                    if path_ext_i == '.cur': #read other *.cur file e.g. tuning
                        filename_com_i = filename_i[:-4]
                        data_type_i = '1D'
                        channel_i = 'Other'
                        data_dict_chan_i = wsxm_readspectra(path_i, all_files=False)
                        header_i = data_dict_chan_i['header']
                        res_i = header_i['Number of points [General Info]']
                        spec_dir_i = list(data_dict_chan_i['data'].keys())
                        size_i = str(data_dict_chan_i['data'][spec_dir_i[0]]['x'].max())  + ' ' + header_i['X axis unit [General Info]']
                        # feedback_i = ''
                        spectrodf_i = convert_spectro2df(data_dict_chan_i['data'])
                        z_max_i = spectrodf_i['y'].max()
                        z_min_i = spectrodf_i['y'].min()
                        z_avg_i = spectrodf_i['y'].mean()
                        fig_i = fig2html(plotly_lineplot(data=spectrodf_i, x="x", y="y", color="segment"), plot_type='plotly')
                        # sns.lineplot(data=spectrodf_i, x="x", y="y", hue="segment")
                        # fig_i = fig2html(plt.gcf())
                        # plt.close()
    
                file_dict['file'].append(filename_com_i)
                file_dict['name'].append(filename_i)
                file_dict['channel'].append(channel_i)
                file_dict['type'].append(data_type_i)
                file_dict['size'].append(size_i)
                file_dict['resolution'].append(res_i)
                file_dict['max'].append(z_max_i)
                file_dict['min'].append(z_min_i)
                file_dict['avg'].append(z_avg_i)
                # file_dict['feedback'].append(feedback_i)
                file_dict['plot'].append(fig_i)
                file_dict['extension'].append(path_ext_i)
                file_dict['time'].append(time_i)
                file_dict['header'].append(header_i)       
                file_dict['header names'].append(list(header_i.keys()))                         
        
        file_df = pd.DataFrame.from_dict(file_dict)
        # print(file_df['header names'].to_numpy().flatten().unique())   
        
        # file_df.drop(columns=['plot', 'header names']).to_excel(f"{folderpath}/filelist_{os.path.basename(folderpath)}.xlsx")
        file_df['header data'] = file_df['header'].map(str) #convert dictionary column data to string for excel saving
        file_df.sort_values(by=['time'], inplace=True, ignore_index=True)
        #save excel file for manual check including images
        imagedf_to_excel(file_df.drop(columns=['header', 'header names']), 
                         f"{folderpath}/filelist_{os.path.basename(folderpath)}.xlsx", img_size=(100, 100))
        file_df.drop(columns=['header data'], inplace=True) #remove "stringed" header data
        #save "pickled" binary data of file list for later use   
        file_df.to_pickle(f"{folderpath}/filelist_{os.path.basename(folderpath)}.pkl")
    
    return file_df