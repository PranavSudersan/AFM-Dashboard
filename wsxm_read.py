import struct
import os
import re
import numpy as np

DATA_TYPES = {'short':(2,'h'),'short-data':(2,'h'), 'unsignedshort':(2,'H'),
              'integer-data':(4,'i'), 'signedinteger':(4,'i'),
              'float-data':(4,'f'), 'double':(8,'d')}

#rename spectroscopy line to standard names: approach and retract
SPECT_DICT = {'Forward':'approach', 'Backward': 'retract',
              'b': 'retract', 'f': 'approach'} 

def wsxm_get_common_files(filepath):
    # filepath = 'data/interdigThiols_tipSi3nN_b_0026.fb.ch1.gsi'
    path_dir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    # filename_com = os.path.basename(filepath).split('.')[0] #common file name
    match = re.search(r'\d{4}', filename) #regex to find 4 digit number in filename
    filename_com = filename[:match.start()+4]
    # print(filename_com)
    files = []
    for i in os.listdir(path_dir):
        path_i = os.path.join(path_dir,i)
        if os.path.isfile(path_i) and i.startswith(filename_com):
            files.append(path_i)    
    
    return files


#read WSxM header data
def wsxm_readheader(file, pos=0, inibyte=100):
    header_dict = {}
    # Find header size
    file.seek(pos, 0)
    data = file.read(inibyte)
    for ln in data.splitlines():
        hd_lst = ln.decode('ascii', errors='ignore').split(':')
        if len(hd_lst) == 2:
            if hd_lst[0] == 'Image header size':
                header_size = int(hd_lst[1])
                # print(header_size)
                break
    # read header data
    file.seek(pos, 0)
    data = file.read(header_size)#[:header_size]
    for ln in data.splitlines():
        hd_lst = ln.decode('ascii', errors='ignore').split(':')
        if len(hd_lst) == 2:
            header_dict[hd_lst[0].strip()] = hd_lst[1].strip()
    
    pos_new = pos + header_size #bytes read so far
    # print(header_dict)
    return header_dict, pos_new

#read WSxM binary image data
def wsxm_readimg(file, header_dict, pos):
    data_format = header_dict['Image Data Type']
    chan_label = header_dict['Acquisition channel']
    line_rate = float(header_dict['X-Frequency'].split(' ')[0])
    x_num = int(header_dict['Number of rows'])
    y_num = int(header_dict['Number of columns'])
    x_len = float(header_dict['X Amplitude'].split(' ')[0])
    y_len = float(header_dict['Y Amplitude'].split(' ')[0])
    z_len = float(header_dict['Z Amplitude'].split(' ')[0])
    x_dir = header_dict['X scanning direction']
    y_dir = header_dict['Y scanning direction'] #CHECK Y DIRECTIONS
    #CHECK THIS FOR SECOND ARRAY! MAY NOT WORK FOR 3D Mode images!
    #THIS DOES NOT WORK. CHECK EVERYWHERE
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
    if z_len == 0: #for zero data
        z_calib = 1
    else:
        z_calib = z_len/(ch_array.max()-ch_array.min())
    
    #img data dictionary
    data_dict_chan = {'data': {'Z': z_calib*ch_array.reshape(x_num, y_num),
                               'X': x_data,
                               'Y': y_data},
                      'header': header_dict}
    
    pos += data_len #bytes read so far
    return data_dict_chan, pos
    
# Read WSxM channel image data
def wsxm_readchan(filepath):
    filepath_all = wsxm_get_common_files(filepath)
    print(filepath_all)
    data_dict = {}
    for path in filepath_all:
        path_ext = os.path.splitext(path)[1] #file extension
        if path_ext != 'gsi': #ignore *.gsi files sharing same name
            file = open(f'{path}','rb')
            header_dict, pos = wsxm_readheader(file)
            chan_label = header_dict['Acquisition channel']
            data_dict_chan, pos = wsxm_readimg(file, header_dict, pos)
            x_dir = header_dict['X scanning direction']
            if chan_label in data_dict.keys():
                data_dict[chan_label][x_dir] = data_dict_chan
            else:
                data_dict[chan_label] = {}
                data_dict[chan_label][x_dir] = data_dict_chan
            file.close()
    return data_dict

# read *.curves file with image and f-d curves
#TODO: read other spectro data (*.stp and *.cur) similarly and output it in the same format as data_dict below!
#TODO: apply Conversion Factor to final channel value. CHECK THIS EVERYWHERE!
def wsxm_readcurves(filepath):
    filepath_all = wsxm_get_common_files(filepath)
    data_dict = {}
    for path in filepath_all:
        path_ext = os.path.splitext(path)[1] #file extension
        if path_ext == '.curves': # read *.curves spectroscopy files
            file = open(f'{path}','rb')
            header_dict, pos = wsxm_readheader(file)
            data_dict_chan, pos = wsxm_readimg(file, header_dict, pos) 
            
            data_format = header_dict['Image Data Type']
            point_length, type_code  = DATA_TYPES[data_format]
            data_dict_curv = {}
            
            while True:
                # file.seek(pos, 0)
                header_dict, pos = wsxm_readheader(file, pos=pos)     
                line_pts = int(header_dict['Number of points'])
                line_num = int(header_dict['Number of lines'])
                y_label = header_dict['Y axis text'].split('[')[0].strip()
                x_label = header_dict['X axis text'].split('[')[0].strip()
                curv_ind = int(header_dict['Index of this Curve'])
                curv_num = int(header_dict['Number of Curves in this serie'])
                #CHECK THIS FOR SECOND ARRAY! MAY NOT WORK FOR 3D Mode!
                # chan_adc2v = 1#20/2**16 #adc to volt converter for 20V DSP, 16 bit resolution
                chan_fact = float(header_dict['Conversion Factor 00'].split(' ')[0])
                chan_offs = float(header_dict['Conversion Offset 00'].split(' ')[0])
                
                aqpt_x, aqpt_y = tuple(map(float, header_dict['Acquisition point'].replace('nm','').
                                           replace('(','').replace(')','').split(',')))
                time_f = float(header_dict['Forward plot total time'].split(' ')[0])
                time_b = float(header_dict['Backward plot total time'].split(' ')[0])
                
                line_order = ['approach', 'retract']
                if header_dict['First Forward'] == 'No': #CHECK THIS
                    line_order = ['retract', 'approach']
        
                data_len = line_pts*line_num*2*point_length
                file.seek(pos, 0)
                bin_data = file.read(data_len)
                ch_array = np.array(list(struct.iter_unpack(f'{type_code}', bin_data))).flatten()
                x_data, y_data = np.split(ch_array[::2], 2), np.split(ch_array[1::2], 2)
                
                data_dict_curv[curv_ind] = {'header': header_dict, 'data': {}}
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
    
    return data_dict

#read *.stp spectroscopy curves
def wsxm_readstp(filepath):
    filepath_all = wsxm_get_common_files(filepath)
    data_dict = {}
    for path in filepath_all:
        path_ext = os.path.splitext(path)[1] #file extension
        if path_ext == '.stp': # read *.stp spectroscopy files
            file = open(f'{path}','rb')
            filename = os.path.basename(path)
            header_dict, pos = wsxm_readheader(file)
            data_format = header_dict['Image Data Type']
            chan_label = filename.split('_')[-1].split('.')[0] #header_dict['Acquisition channel']
            # line_rate = float(header_dict['X-Frequency'].split(' ')[0])
            x_num = int(header_dict['Number of rows'])
            y_num = int(header_dict['Number of columns'])
            x_len = float(header_dict['X Amplitude'].split(' ')[0])
            y_len = float(header_dict['Y Amplitude'].split(' ')[0])
            z_len = float(header_dict['Z Amplitude'].split(' ')[0])
            x_dir = header_dict['X scanning direction']
            y_dir = header_dict['Y scanning direction'] #CHECK Y DIRECTIONS
            z_dir = SPECT_DICT[filename.split('.')[-2]]
            chan_fact = float(header_dict['Conversion Factor 00'].split(' ')[0])
            chan_offs = float(header_dict['Conversion Offset 00'].split(' ')[0])

            z_data = np.linspace(x_len, 0, y_num, endpoint=True) #CHECK THIS
            
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
                z_calib = z_len/(ch_array.max()-ch_array.min())
        
            for i in range(x_num):
                curv_ind = i + 1
                #data dictionary initialised in a consistant format (also check wsxm_readcurves())
                if chan_label not in data_dict.keys():
                    data_dict[chan_label] = {'curves': {}, 'image':{}}
                if curv_ind not in data_dict[chan_label]['curves'].keys():
                    data_dict[chan_label]['curves'][curv_ind] = {'data': {},'header': header_dict}
                if z_dir not in data_dict[chan_label]['curves'][curv_ind]['data'].keys():
                    data_dict[chan_label]['curves'][curv_ind]['data'][z_dir] = {}
                data_dict[chan_label]['curves'][curv_ind]['data'][z_dir]['x'] = z_data
                data_dict[chan_label]['curves'][curv_ind]['data'][z_dir]['y'] = z_calib*ch_mat[:][i] #chan_offs+(ch_mat[:][i]*chan_fact)

            file.close()
    
    return data_dict


# Read WSxM Force volume data
def wsxm_readforcevol(filepath):
    filepath_all = wsxm_get_common_files(filepath)
    data_dict = {}
    for path in filepath_all:
        path_ext = os.path.splitext(path)[1] #file extension
        # if path_ext == '.top': #topgraphy data
        #     data_dict['Topography'] = wsxm_readchan(path)
        if path_ext == '.gsi': #force volume data from *.gsi files
            file = open(f'{path}','rb')
            header_dict, pos = wsxm_readheader(file)
            
            data_format = header_dict['Image Data Type']
            chan_label = header_dict['Acquisition channel']
            spec_dir = header_dict['Spectroscopy type']
            x_dir = spec_dir.split(' ')[1]
            y_dir = header_dict['Y scanning direction'] #CHECK Y DIRECTIONS
            line_rate = float(header_dict['X-Frequency'].split(' ')[0])
            x_num = int(header_dict['Number of rows'])
            y_num = int(header_dict['Number of columns'])
            chan_num = int(header_dict['Number of points per ramp'])
            x_len = float(header_dict['X Amplitude'].split(' ')[0])
            y_len = float(header_dict['Y Amplitude'].split(' ')[0])
            z_len = float(header_dict['Z Amplitude'].split(' ')[0])
            chan_adc2v = float(header_dict['ADC to V conversion factor'].split(' ')[0])
            chan_fact = float(header_dict['Conversion factor 0 for input channel'].split(' ')[0])
            chan_offs = float(header_dict['Conversion offset 0 for input channel'].split(' ')[0])
                    
            x_data = np.linspace(0, x_len, x_num, endpoint=True) if x_dir == 'Backward' else np.linspace(x_len, 0, x_num, endpoint=True)
            y_data = np.linspace(0, y_len, y_num, endpoint=True) if y_dir == 'Up' else np.linspace(y_len, 0, y_num, endpoint=True)
            # xx_data, yy_data = np.meshgrid(x_data, y_data)
        
            z_data = np.empty(0)
            for i in range(chan_num):
                z_data = np.append(z_data, float(header_dict[f'Image {i:03}'].split(' ')[0]))
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
            topo_calib = z_len/(topo_array.max()-topo_array.min())
            #topo data dictionary
            data_dict_topo = {'data': {'Z': topo_calib*topo_array.reshape(x_num, y_num),
                                       'X': x_data,
                                       'Y': y_data
                                       },
                              'header': header_dict}
            topo_label = 'Topography'
            if topo_label not in data_dict.keys():
                data_dict[topo_label] = {}
            data_dict[topo_label][spec_dir] = data_dict_topo
            
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
    return data_dict