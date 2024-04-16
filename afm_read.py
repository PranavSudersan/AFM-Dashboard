import numpy as np
import pandas as pd
import struct
import zipfile
import tifffile as tiff
from igor import binarywave
# import time

class JPKRead:        

    DATA_TYPES = {'short':(2,'h'),'short-data':(2,'h'), 'unsignedshort':(2,'H'),
                  'integer-data':(4,'i'), 'signedinteger':(4,'i'),
                  'float-data':(4,'f')}
    #anal_dict: ANALYSIS_MODE_DICT[mode] dictionary reference
    def __init__(self, file_path, anal_dict, segment_path, jump_tol=0.8):
        self.file_path = file_path
        self.anal_dict = anal_dict
        self.segment_path = segment_path
        self.df = {}
        self.file_format = self.file_path.split('.')[-1]
        if self.file_format == 'jpk-qi-data':
            modes = ['Adhesion', 'Snap-in distance']
        elif self.file_format == 'jpk-force':
            modes = ['Force-distance']
        elif self.file_format == 'jpk':
            modes = ['Height (measured)']   
        if self.file_format == 'jpk':
            self.jpk_data_dict = self.read_jpk(self.file_path)
            for mode in modes:
                result = self.anal_dict[mode]['function'](channel=mode,
                                                          trace=self.anal_dict[mode]['misc']['trace'],
                                                          calibration=self.anal_dict[mode]['misc']['calibration']) 
#                 result = self.anal_dict[mode]['function'](dirpath=self.segment_path)
                for key, value in result.items():
                    output = self.anal_dict[mode]['output']
                    output[key] = np.append(output[key], value)
                self.df[mode] = pd.DataFrame(self.anal_dict[mode]['output'])
#             self.get_height_measured(file_path, modes)
        elif self.file_format == 'ibw':
            self.ibw_data_dict = self.read_ibw(self.file_path)
            if self.ibw_data_dict['header']['raw data type'] == 'force':
                modes = ['Force-distance']
            else:
                modes = ['Height (measured)']
            for mode in modes:
                result = self.analyze_ibw(mode=mode) 
#                 result = self.anal_dict[mode]['function'](dirpath=self.segment_path)
                for key, value in result.items():
                    output = self.anal_dict[mode]['output']
                    output[key] = np.append(output[key], value)
                self.df[mode] = pd.DataFrame(self.anal_dict[mode]['output'])
        elif self.file_format in ['ch15', 'ch16']:
            result = self.read_wsxm_chan(self.file_path)
        else:
            self.data_zip = zipfile.ZipFile(self.file_path, 'r')
            self.get_data(modes, jump_tol=jump_tol)

 #read WsxM file
    def read_wsxm_chan(self, filepath):
        pass
        
    #read jpk TIFF file and return data matrixes (check JPK TIFF specification for hex code details)
    def read_jpk(self, filepath):
        with tiff.TiffFile(filepath) as tif:
            data_dict = {}
            for page in tif.pages:        
                tag_dict = {}
                for tag in page.tags:

                    tag_formatted = hex(tag.code) if len(tag.name)==5 else tag.name
        #             print(tag_formatted,tag.name,tag.value)
                    tag_dict[tag_formatted] = tag.value

                image_array = page.asarray()

                if tag_dict[hex(0x8050)] == 'thumbnail':
                    feedback_prop = tag_dict[hex(0x803E)]
                    feedback_dict = {}
                    for k in feedback_prop.splitlines():
                        k_split = k.split(':')
                        feedback_dict[k_split[0].strip()] = k_split[1].strip()
        #             data_dict['Feedback properties'] = feedback_dict

                    x0 = tag_dict[hex(0x8040)]
                    y0 = tag_dict[hex(0x8041)]
                    x_len = tag_dict[hex(0x8042)]
                    y_len = tag_dict[hex(0x8043)]
                    scan_angle = tag_dict[hex(0x8044)]
                    grid_reflect = tag_dict[hex(0x8045)]
                    x_num = tag_dict[hex(0x8046)]
                    y_num = tag_dict[hex(0x8047)]            

                    data_dict['header'] = {'Feedback_Mode': tag_dict[hex(0x8030)],
                                           'Grid-x0':x0,
                                           'Grid-y0':y0,
                                           'Grid-uLength':x_len,
                                           'Grid-vLength':y_len,
                                           'Grid-Theta':scan_angle,
                                           'Grid-Reflect':grid_reflect,
                                           'Grid-iLength':x_num,
                                           'Grid-jLength':y_num,
                                           'Lineend':tag_dict[hex(0x8048)],
                                           'Scanrate-Frequency':tag_dict[hex(0x8049)],
                                           'Scanrate-Dutycycle':tag_dict[hex(0x804A)],
                                           'Motion':tag_dict[hex(0x804B)],
                                           'Feedback properties':feedback_dict}

                    x_data = np.linspace(x0, x0+x_len, num=x_num)
                    y_data = np.linspace(y0, y0+y_len, num=y_num)
                    xx_data, yy_data = np.meshgrid(x_data, y_data)
                else:
                    channel_name = tag_dict[hex(0x8052)]
                    channel_trace = 'trace' if tag_dict[hex(0x8051)] == 0 else 'retrace'
                    if channel_name not in data_dict.keys():
                        data_dict[channel_name] = {}
                    data_dict[channel_name][channel_trace] = {}
                    num_slots = tag_dict[hex(0x8080)]
                    for i in range(num_slots):
                        slot_name = tag_dict[hex(0x8090 + i*0x30)]
                        slot_type = tag_dict[hex(0x8091 + i*0x30)]
                        calibration_name = tag_dict[hex(0x80A0 + i*0x30)]
                        data_dict[channel_name][channel_trace][slot_name] = {}
                        if 'absolute' in slot_type.lower():
                            data_dict[channel_name][channel_trace][slot_name]['data'] = image_array
                            unit = ''
                        elif 'relative' in slot_type.lower():
                            slot_parent = 'raw'#tag_dict[hex(0x8092 + i*0x30)]
                            unit = tag_dict[hex(0x80A2 + i*0x30)]
                            scaling_type = tag_dict[hex(0x80A3 + i*0x30)]
                            if 'linear' in scaling_type.lower():
                                multiplier = tag_dict[hex(0x80A4 + i*0x30)]
                                offset = tag_dict[hex(0x80A5 + i*0x30)]
                                data_dict[channel_name][channel_trace][slot_name]['data'] = offset + \
                                    (multiplier*data_dict[channel_name][channel_trace][slot_parent]['data'])
                        else:
                            continue
                        data_dict[channel_name][channel_trace][slot_name]['info'] = {'Calibration name': calibration_name,
                                                                                    'Unit': unit}
        return data_dict

                
    #import datafile and get output dataframe
    def get_data(self, modes, jump_tol, *args, **kwargs):
        #print(self.segment_path)
##        self.file_format = self.file_path.split('.')[-1]
##        self.data_zip = zipfile.ZipFile(self.file_path, 'r')
        #with zipfile.ZipFile(self.file_path, 'r') as self.data_zip:
        shared_header = self.data_zip.read('shared-data/header.properties').decode().split('\n')
        self.shared_header_dict = self.parse_header_file(shared_header)    
        file_list = self.data_zip.namelist()
        
        if self.segment_path == None: #all segments taken
            for file in file_list:
                if file.endswith('segments/'): # segments folder
                    for mode in modes:
                        result = self.anal_dict[mode]['function'](dirpath=file, jump_tol=jump_tol)
                        for key, value in result.items():
                            output = self.anal_dict[mode]['output']
                            output[key] = np.append(output[key], value)
        else: #specific segment taken
            for mode in modes:
                result = self.anal_dict[mode]['function'](dirpath=self.segment_path)
                for key, value in result.items():
                    output = self.anal_dict[mode]['output']
                    output[key] = np.append(output[key], value)

        for mode in modes:
            self.df[mode] = pd.DataFrame(self.anal_dict[mode]['output'])

    def parse_header_file(self, header):
        header_dict = {}
        header_dict['Date'] = header[0]
        for line in header[1:]:
            if line != '':
                line_data = line.split('=')
                keys = line_data[0].split('.')
                value = line_data[1]
                temp_dict = header_dict
                for idx, key in enumerate(keys):        
                    if key not in temp_dict.keys():
                        if idx == len(keys)-1:
                            temp_dict[key] = value
                        else:
                            temp_dict[key] = {}
                    temp_dict = temp_dict[key]
        return header_dict

    def decode_data(self, channel, segment_header_dict, segment_dir):
        info_id = segment_header_dict['channel'][channel]['lcd-info']['*']
        decode_dict = self.shared_header_dict['lcd-info'][info_id]
        data = self.data_zip.read(f'{segment_dir}/channels/{channel}.dat')
        
        point_length, type_code  = self.DATA_TYPES[decode_dict['type']]
        num_points = len(data) // point_length
        data_unpack = np.array(struct.unpack_from(f'!{num_points}{type_code}', data))
        encod_multiplier = float(decode_dict['encoder']['scaling']['multiplier'])
        encod_offset = float(decode_dict['encoder']['scaling']['offset'])
        data_conv = (data_unpack * encod_multiplier) + encod_offset
        conv_list = decode_dict['conversion-set']['conversions']['list'].split(' ')
        data_dict = {} #data dictionary of each level of converted data (eg. nominal/calibrated, distance/force)
        for conv in conv_list:
            multiplier = float(decode_dict['conversion-set']['conversion'][conv]['scaling']['multiplier'])
            offset = float(decode_dict['conversion-set']['conversion'][conv]['scaling']['offset'])
            data_conv = (data_conv * multiplier) + offset
            data_dict[conv] = data_conv

        return data_dict
        
    #Reference code: https://github.com/AFM-analysis/afmformats/blob/master/afmformats/formats/fmt_igor.py
    def read_ibw(self, filepath):
        ibw = binarywave.load(filepath)
        wdata = ibw["wave"]["wData"]
        notes = {}
        for line in str(ibw["wave"]["note"]).split("\\r"):
            if line.count(":"):
                key, val = line.split(":", 1)
                try:
                    notes[key] = float(val.strip())
                except ValueError:
                    notes[key] = val.strip()
        
        #CHECK THIS! MIGHT NOT WORK FOR SOME DATA
        if 'ForceDist' in notes.keys():
            notes['raw data type'] = 'force'
        else:
            notes['raw data type'] = 'image'
            
        for ll in ibw["wave"]["labels"]:
            for li in ll:
                if li:
                    print(li.decode())
#         print(ibw['wave'].keys())
#         print(ibw['wave']["wave_header"])
#         print(notes)
#         print(wdata)
#         print(wdata[0].shape)
        # Metadata
#         metadata = {}
#         # acquisition
#         for k,v in notes.items():
#             metadata[k] = v
        
        # Data
        labels = []
        data = {}
        ind = 0 
        for ll in ibw["wave"]["labels"]:
            for li in ll:
                if li:
                    labelname = li.decode()
                    data[labelname] = wdata[..., ind]
                    ind += 1
#         print(wdata.shape, len(data.keys()))
#         print(data)
        assert len(data.keys()) == wdata.shape[-1]

        for fkey in ["Defl", "Deflection"]:
            if fkey in data.keys():
                # force is in [m] (convert to [N])
                data["force"] = data["Defl"] \
                    * notes['SpringConstant']
                break
        dataset = {"data": data,
                   "header": notes,
                  }
        return dataset
    
    def analyze_ibw(self, mode):
        ibw_data_dict = self.ibw_data_dict
        if mode == 'Height (measured)':
            trace = self.anal_dict[mode]['misc']['trace']
            if trace == 'average':
                z_data = np.mean([ibw_data_dict['data']['HeightTrace'],
                                  ibw_data_dict['data']['HeightRetrace']], axis=0)
            elif trace == 'trace':
                z_data = ibw_data_dict['data']['HeightTrace']
            elif trace == 'retrace':
                z_data = ibw_data_dict['data']['HeightRetrace']
            
            x0 = ibw_data_dict['header']['XOffset']
            y0 = ibw_data_dict['header']['YOffset']
            x_len = ibw_data_dict['header']['FastScanSize']
            y_len = ibw_data_dict['header']['SlowScanSize']
            x_num = int(ibw_data_dict['header']['ScanPoints'])
            y_num = int(ibw_data_dict['header']['ScanLines'])
            scan_angle = ibw_data_dict['header']['ScanAngle']
            x_data = np.linspace(x0, x0+x_len, num=x_num)
            y_data = np.linspace(y0, y0-y_len, num=y_num) #CHECK THIS
            xx_data, yy_data = np.meshgrid(x_data, y_data)
    #         mode = modes[0] #CHECK
            self.rotation_info = [x0, y0, scan_angle]
    #         start = time.process_time()
    #         print(x_num,y_num)

            #USE SAME KEYS AS ANALYSIS_MODE_DICT!
            result_dict = {'Height': np.array(z_data).flatten(),
                           'X': xx_data.flatten(), 
                           'Y':yy_data.flatten(),
                           'Segment folder': [None]*(x_num*y_num)}
            
        elif mode == 'Force-distance':
            #USE SAME KEYS AS ANALYSIS_MODE_DICT
            result_dict = {'Force': [], 'Measured height': [], 'Distance': [], 'Segment': [],
                           'Segment folder': [], 'X': [], 'Y': []}
            #TODO: get segment info from header files
            idx_max = np.argmax(ibw_data_dict['data']['force'])
            segment_slices = {'extend':slice(0,idx_max), 
                              'retract':slice(idx_max,len(ibw_data_dict['data']['force']))}
            print('slice', segment_slices)
            for seg_name, seg_slice in segment_slices.items():        
                #get segment force data
                force_data = ibw_data_dict['data']['force'][seg_slice]
                #get piezo measured height data
                height_data = -ibw_data_dict['data']['ZSnsr'][seg_slice]
                #get cantilever deflection
                defl_data = ibw_data_dict['data']['Defl'][seg_slice]
                #tip sample distance
                distance_data = height_data + (defl_data)#-defl_data[0]

                #get position
                x_pos, y_pos = ibw_data_dict['header']['XLVDT'], ibw_data_dict['header']['YLVDT']

                result_dict['Force'] = np.append(result_dict['Force'],force_data)
                result_dict['Measured height'] = np.append(result_dict['Measured height'],height_data)
                result_dict['Distance'] = np.append(result_dict['Distance'],distance_data)
                len_data = len(force_data)
                result_dict['Segment'] = np.append(result_dict['Segment'],
                                                   len_data * [seg_name])
                result_dict['Segment folder'] = np.append(result_dict['Segment folder'],
                                                          len_data * [None])
                result_dict['X'] = np.append(result_dict['X'],
                                             len_data * [x_pos])
                result_dict['Y'] = np.append(result_dict['Y'],
                                             len_data * [y_pos])
        
        return result_dict
