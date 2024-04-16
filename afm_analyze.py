# include functions to analyze and get output data from afm data

import numpy as np
import pandas as pd
import itertools
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans
import statistics
from matplotlib.widgets import Slider

from afm_read import JPKRead
from afm_plot import AFMPlot

class JPKAnalyze(JPKRead):

    #mapping of keys in header files which depend on file format
    FORMAT_KEY_DICT = {'jpk-qi-data': {'position': 'position'},
                       'jpk-force': {'position': 'start-position'}}
    def __init__(self, file_path, segment_path, jump_tol=0.8):
        #Make sure variable keys in result dict of 'function' definition
        #is same as 'output' keys of below
        #TODO: simplify dictionary to automatically include all possible channel pre-existing in afm file
        self.ANALYSIS_MODE_DICT = {'Adhesion': {'function': self.get_adhesion,
                                                'output': {'Adhesion': [],'X': [], 'Y':[],
                                                           'Segment folder': []},
                                                'plot_parameters': {'x': 'X',
                                                                    'y': 'Y',
                                                                    'z': 'Adhesion',
                                                                    'title': 'Adhesion',
                                                                    'type': ['2d'],
                                                                    'points_flag':False}
                                                },
                                   'Snap-in distance': {'function': self.get_snapin_distance,
                                                        'output': {'Height': [],
                                                                   'X': [], 'Y':[],
                                                                   'Segment folder': []},
                                                        'plot_parameters': {'x': 'X',
                                                                            'y': 'Y',
                                                                            'z': 'Height',
                                                                            'title': 'Jump-in distance',
                                                                            'type': ['2d'],
                                                                            'points_flag':False}
                                                        },
                                   'Height (measured)': {'function': self.get_height_measured,
                                                        'output': {'Height': [],
                                                                   'X': [], 'Y':[],
                                                                   'Segment folder': []},
                                                        'plot_parameters': {'x': 'X',
                                                                            'y': 'Y',
                                                                            'z': 'Height',
                                                                            'title': 'Height (measured)',
                                                                            'type': ['2d'],
                                                                            'points_flag':True},
                                                         'misc': {'trace': 'retrace',#trace, retrace, average
                                                                  'calibration': 'nominal'}#nominal,degrees
                                                        },
                                   'Force-distance': {'function': self.get_force_distance,
                                                      'output': {'Force': [],
                                                                 'Measured height': [],
                                                                 'Distance': [],
                                                                 'X': [], 'Y':[],
                                                                 'Segment': [],
                                                                 'Segment folder': []},
                                                      'plot_parameters': {'x': 'Distance',
                                                                          'y': 'Force',
                                                                          'style': 'Segment',
                                                                          'title': 'Force-distance curve',
                                                                          'type': ['line'],
                                                                          'points_flag':False}
                                                      },
                                   'Misc': {'figure_list': []}
                                   }
        #initialize JPKRead and get data
        super().__init__(file_path, self.ANALYSIS_MODE_DICT,
                         segment_path, jump_tol=jump_tol)

    #clear output data in ANALYSIS_MODE_DICT
    def clear_output(self, mode):
        for key in self.ANALYSIS_MODE_DICT[mode]['output'].keys():
            self.ANALYSIS_MODE_DICT[mode]['output'][key] = []
    
    #get channel data from .jpk file
    def get_height_measured(self, channel, trace, calibration):

        jpk_data_dict = self.jpk_data_dict
        if trace == 'average':
            z_data = np.mean([jpk_data_dict[channel]['trace'][calibration]['data'],
                             jpk_data_dict[channel]['retrace'][calibration]['data']], axis=0)
        else:
            z_data = jpk_data_dict[channel][trace]['nominal']['data']
        #print(matlab_output['header'], matlab_output.keys())
        x0 = jpk_data_dict['header']['Grid-x0']
        y0 = jpk_data_dict['header']['Grid-y0']
        x_len = jpk_data_dict['header']['Grid-uLength']
        y_len = jpk_data_dict['header']['Grid-vLength']
        x_num = int(jpk_data_dict['header']['Grid-iLength'])
        y_num = int(jpk_data_dict['header']['Grid-jLength'])
        scan_angle = jpk_data_dict['header']['Grid-Theta']
        x_data = np.linspace(x0, x0+x_len, num=x_num)
        y_data = np.linspace(y0, y0+y_len, num=y_num)
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
#         output = self.anal_dict[mode]['output']
        
#         output['Height'] = np.array(z_data).flatten()
#         output['X'] = xx_data.flatten()
#         output['Y'] = yy_data.flatten()
#         output['Segment folder']= [None]*(x_num*y_num)
        
#         for i in range(x_num-1):
#             for j in range(y_num-1):
#                 #x_rotated = x0 + (x_data[i]-x0)*np.cos(scan_angle) + (y_data[j]-y0)*np.sin(scan_angle)
#                 #y_rotated = y0 -(x_data[i]-x0)*np.sin(scan_angle) + (y_data[j]-y0)*np.cos(scan_angle)
#                 output = self.anal_dict[mode]['output']
#                 output['Height'] = np.append(output['Height'], z_data[j][i])
#                 output['X'] = np.append(output['X'], x_data[i])
#                 output['Y'] = np.append(output['Y'], y_data[j])
#                 output['Segment folder'].append(None) #CHECK
#         print(time.process_time() - start)
#         self.df[mode] = pd.DataFrame(self.anal_dict[mode]['output'])
        return result_dict
    
    def get_adhesion(self, dirpath, *args, **kwargs):    
        retract_dir = f'{dirpath}1'  #retract folder  
        segment_header_path = f'{retract_dir}/segment-header.properties'
        segment_header = self.data_zip.read(segment_header_path).decode().split('\n')
        segment_header_dict = self.parse_header_file(segment_header)
        #get retract force data
        force_data = self.decode_data('vDeflection', segment_header_dict,
                                      retract_dir)['force']
        #calculate adhesion
        adhesion = force_data[-1] - force_data.min()

        #get position
        x_pos, y_pos = self.get_xypos(segment_header_dict)

        #USE SAME KEYS AS ANALYSIS_MODE_DICT!
        result_dict = {'Adhesion': adhesion,
                       'X': x_pos, 'Y':y_pos,
                       'Segment folder': dirpath} 
        return result_dict

    def get_snapin_distance(self, dirpath, jump_tol=0.8, *args, **kwargs):    
        extend_dir = f'{dirpath}0'  #extend folder
        #get segment header file
        segment_header_path = f'{extend_dir}/segment-header.properties'
        segment_header = self.data_zip.read(segment_header_path).decode().split('\n')
        segment_header_dict = self.parse_header_file(segment_header)
        #get extend force data
        force_data = self.decode_data('vDeflection', segment_header_dict,
                                      extend_dir)['force']
        #get extend height data
        height_data = self.decode_data('measuredHeight', segment_header_dict,
                                       extend_dir)['nominal']

        #calculate snapin distance
        force_sobel = ndimage.sobel(force_data) #sobel transform
        idx_min = np.argmin(force_sobel)
##        idx_min = np.argmin(force_data) #minimum force during extend
        #tolerance method to get jumpin distance
##        zero_points = 10
##        zero_force = statistics.median(force_data[:zero_points])
##        zero_dev = statistics.stdev(force_data[:zero_points])
##        tol = 100 #deviation from zero
##        for idx, a in enumerate(force_data): 
##            if abs(a-zero_force) > tol*zero_dev:
##                idx_min = idx
##                #idx_min = force_data.index(a)
##                #idx, = np.where(force_data == a)
##                #print(idx)
##                #idx_min = idx[0]
##                break
##            else:
##                idx_min = len(force_data)-1
        #print(idx_min)
        snapin_distance = height_data[idx_min] - height_data[-1]
        #TODO: define as fraction of extend distance range
        total_distance = height_data[0] - height_data[-1]
        #tolerence = 0.8
        snapin_distance = 0 if snapin_distance >= jump_tol * total_distance \
                          else snapin_distance

        #get position
        x_pos, y_pos = self.get_xypos(segment_header_dict)
        
        adhesion = force_data[-1] - force_data.min()
        #USE SAME KEYS AS ANALYSIS_MODE_DICT!
        result_dict = {'Height': snapin_distance,
                       'X': x_pos, 'Y':y_pos,
                       'Segment folder': dirpath}
        return result_dict

    def get_force_distance(self, dirpath, *args, **kwargs):
        #USE SAME KEYS AS ANALYSIS_MODE_DICT
        result_dict = {'Force': [], 'Measured height': [], 'Distance': [], 'Segment': [],
                       'Segment folder': [], 'X': [], 'Y': []}
        #TODO: get segment info from header files
        segment_name = {'0': 'extend', '1': 'retract'}
        for seg_num, seg_name in segment_name.items():        
            #get segment header file
            segment_dir = f'{dirpath}{seg_num}'  #extend folder
            segment_header_path = f'{segment_dir}/segment-header.properties'
            segment_header = self.data_zip.read(segment_header_path).decode().split('\n')
            segment_header_dict = self.parse_header_file(segment_header)
            #get segment force data
            force_data = self.decode_data('vDeflection', segment_header_dict,
                                          segment_dir)['force']#*1.87920/1.08133
            #get piezo measured height data
            height_data = self.decode_data('measuredHeight', segment_header_dict,
                                           segment_dir)['nominal']
            #get cantilever deflection
            defl_data = self.decode_data('vDeflection', segment_header_dict,
                                         segment_dir)['distance']#*33.6146/46.6018
            #tip sample distance
            distance_data = height_data + (defl_data)#-defl_data[0]

            #get position
            x_pos, y_pos = self.get_xypos(segment_header_dict)
            
            result_dict['Force'] = np.append(result_dict['Force'],force_data)
            result_dict['Measured height'] = np.append(result_dict['Measured height'],height_data)
            result_dict['Distance'] = np.append(result_dict['Distance'],distance_data)
            len_data = len(force_data)
            result_dict['Segment'] = np.append(result_dict['Segment'],
                                               len_data * [seg_name])
            result_dict['Segment folder'] = np.append(result_dict['Segment folder'],
                                                      len_data * [dirpath])
            result_dict['X'] = np.append(result_dict['X'],
                                         len_data * [x_pos])
            result_dict['Y'] = np.append(result_dict['Y'],
                                         len_data * [y_pos])

        return result_dict

    def get_xypos(self, segment_header_dict): #get xy position    
        pos_dict = segment_header_dict['force-segment-header']\
                   ['environment']['xy-scanner-position-map']['xy-scanner']\
                   ['tip-scanner'][self.FORMAT_KEY_DICT[self.file_format]['position']]
        return float(pos_dict['x']), float(pos_dict['y'])
    

class DataFit:
    def __init__(self, jpk_anal, mode, func, img_anal,
                 guess=None, bounds=(-np.inf, np.inf), zero=0,
                 output_path=None):
        FIT_DICT = {'Sphere-RC': {'function': self.sphere_rc,
                                  'params': 'R,c'
                                  },
                    'Sphere': {'function': self.sphere,
                                  'params': 'R,c,x0,y0'
                              }
                    } #'func' arg keys and params
##        df_filt = jpk_anal.df.query(filter_string)
        coords = img_anal.coords
        bbox = img_anal.bbox
##        mode = 'Snap-in distance'
        plot_params =  jpk_anal.anal_dict[mode]['plot_parameters']
        x = plot_params['x']
        y = plot_params['y']
        z = plot_params['z']
        df_data =  jpk_anal.df[mode].pivot_table(values=z, index=y, columns=x,
                                                      aggfunc='first')
        self.fit_output = {}
        self.fit_data_full = {}#pd.DataFrame()
        num_fit = 20 #number of points in fit
        for key, val in coords.items():
            data = np.array([[df_data.columns[coord[1]],
                             df_data.index[coord[0]],
                             df_data.iloc[coord[0], coord[1]]] for coord in val])
##        data = np.array([[df_filt[x][i],
##                          df_filt[y][i],
##                          df_filt[z][i]] for i in df_filt.index])
##            print(data.shape, len(data))
            if len(data) > 9:
                x_start  = min([bbox[key][0],bbox[key][2]])
                x_end = max([bbox[key][0],bbox[key][2]])                
                y_start  = min([bbox[key][1],bbox[key][3]])
                y_end = max([bbox[key][1],bbox[key][3]])                
                
                #TODO: get below directly from sphere_rc function
                i, j, k = np.argmax(data, axis=0)
                a, b, c = data[k, 0], data[k, 1], data[k, 2]

                base_r = min([x_end-x_start, y_end-y_start])/2
                h_max = c-zero
                #https://mathworld.wolfram.com/SphericalCap.html
                R_guess = (base_r**2 + h_max**2)/(2*h_max)
                x0_guess = 0.5*(x_start+x_end)
                y0_guess = 0.5*(y_start+y_end)
                
##                R_guess = min(map(abs,[bbox[key][0]-bbox[key][2],
##                                       bbox[key][1]-bbox[key][3]]))
                guess = {'Sphere-RC': [R_guess, -R_guess],
                         'Sphere': [R_guess, -R_guess, x0_guess, y0_guess]
                        }
                #fit
                popt, _ = curve_fit(FIT_DICT[func]['function'], data, data[:,2],
                                    p0=guess[func], bounds=bounds)

                #TODO: weed out bad fits
                self.fit_output[key] = dict(zip(FIT_DICT[func]['params'].split(','), popt))
##                print(key, popt, R_guess, c, h_max)
                #get fitted data
    ##            data_full = np.array([[jpk_anal.df[mode][x][i],
    ##                                   jpk_anal.df[mode][y][i],
    ##                                   jpk_anal.df[mode][z][i]] for i in jpk_anal.df[mode].index])
                
                
##                x_step = (x_end-x_start)/num_fit
##                y_step = (y_end-y_start)/num_fit
##                print(bbox[key])

                h_fit = popt[0] + popt[1] - zero
                base_r_fit = (h_fit*((2*popt[0])-h_fit))**0.5
                self.fit_output[key]['h_fit'] = h_fit #fitted height
                self.fit_output[key]['base_r_fit'] = base_r_fit #fitted contact radius
                if len(popt) == 4:
                    a,b = popt[2], popt[3]
                
                x_fit, y_fit = np.mgrid[a-base_r_fit:a+base_r_fit:complex(0,num_fit),
                                        b-base_r_fit:b+base_r_fit:complex(0,num_fit)]
##                x_fit, y_fit = np.mgrid[x_start:x_end:complex(0,num_fit),
##                                        y_start:y_end:complex(0,num_fit)]
##                print(x_fit.shape, y_fit.shape)
##                print(a, 0.5*(x_start+x_end), b, 0.5*(y_start+y_end))
                z_fit = popt[1] + (abs((popt[0]**2)-((x_fit-a)**2)-((y_fit-b)**2)))**0.5
##                print('max', z_fit.max(), data[k, 2])
##                z_fit = FIT_DICT[func]['function'](data, *popt)
##                fit_data = pd.DataFrame({x: x_fit.flatten(),
##                                         y: y_fit.flatten(),
####                                         f'{z}_raw': data[:,2],
##                                         f'{z}_fit': z_fit.flatten()})
                fit_data = {x: x_fit, y: y_fit, z: z_fit}
##                fit_data['label'] = key
##                self.fit_data_full = self.fit_data_full.append(fit_data)
                self.fit_data_full[key] = fit_data
##                self.fit_output[key]['z_max'] = z_fit.max() #maximum fitted z

##        print(f'Fit output {func}:', self.fit_output)
        #zero height plane
        x_zero, y_zero = np.mgrid[min(df_data.columns):max(df_data.columns):complex(0,num_fit),
                                  min(df_data.index):max(df_data.index):complex(0,num_fit)]
        z_zero = 0*x_zero + zero
        self.fit_data_full['zero'] = {x: x_zero, y: y_zero, z: z_zero}
        #plot
        afm_plot = AFMPlot()
        self.fig = afm_plot.plot_2dfit(self.fit_data_full, df_data, plot_params,
                                       file_path=output_path)

    def sphere_rc(self, X, R, C): #sphere function (only R and C)
        i, j, k = np.argmax(X, axis=0)
        a, b = X[k, 0], X[k, 1]
        x, y, z = X.T
        return C + (abs((R**2)-((x-a)**2)-((y-b)**2)))**0.5
    
    def sphere(self, X, R, C, x0, y0): #sphere function
        x, y, z = X.T
        return C + (abs((R**2)-((x-x0)**2)-((y-y0)**2)))**0.5

#analyze processed data from JPKRead

class DataAnalyze:
    def __init__(self, jpk_anal, mode):
        self.plot_params =  jpk_anal.anal_dict[mode]['plot_parameters']        
        self.df = jpk_anal.df[mode].copy()
        
        self.plot_x = self.plot_params['x']
        self.plot_y = self.plot_params['y']
        self.plot_z = self.plot_params['z']
        #organize data into matrix for heatmap plot
        self.df_matrix = self.df.pivot_table(values=self.plot_z,
                                             index=self.plot_y,
                                             columns=self.plot_x,
                                             aggfunc='first')
        
    def remove_spikes(self, window=3):
        self.df_matrix = self.df_matrix.rolling(window, axis=0, min_periods=0, center=True).median()
        df_temp = self.df_matrix.reset_index().melt(id_vars = self.plot_y, 
                                                    var_name=self.plot_x, value_name=self.plot_z+' corrected')
        self.df = df_temp.merge(self.df.drop(self.plot_z+' corrected', axis=1), 
                                on=[self.plot_x, self.plot_y], how='left')

    #generate Matrix to use with lstsq for levelling
    def poly_matrix(self, x, y, order=2):
        ncols = (order + 1)**2
        G = np.zeros((x.size, ncols))
        ij = itertools.product(range(order+1), range(order+1))
        for k, (i, j) in enumerate(ij):
            G[:, k] = x**i * y**j
        return G


    def level_data(self, points, order=1):
        X,Y = np.meshgrid(self.df_matrix.columns,
                          self.df_matrix.index)

        if order == 1:
            # best-fit linear plane
            A = np.c_[points[:,0], points[:,1], np.ones(points.shape[0])]
            C,_,_,_ = np.linalg.lstsq(A, points[:,2], rcond=None)    # coefficients
            #print(C)
            # evaluate it on grid
##            Z = C[0]*X + C[1]*Y + C[2]
##            print(Z)
            self.df['Zero fit'] = C[0]*self.df[self.plot_x] + \
                                  C[1]*self.df[self.plot_y] + C[2]
            #print(self.df)
        elif order == 2:
            x, y, z = points.T
            #x, y = x - x[0], y - y[0]  # this improves accuracy

            # make Matrix:
            G = self.poly_matrix(x, y, order)
            # Solve for np.dot(G, m) = z:
            m = np.linalg.lstsq(G, z, rcond=None)[0]
            #print('m', m)
            # Evaluate it on a grid...
##            GG = self.poly_matrix(X.ravel(), Y.ravel(), order)
##            Z = np.reshape(np.dot(GG, m), X.shape)
##            print(Z)
            self.df['Zero fit'] = np.polynomial.polynomial.polyval2d(self.df[self.plot_x],
                                                                     self.df[self.plot_y],
                                                                     np.reshape(m, (-1, 3)))
        
        self.df[self.plot_z+' corrected'] = self.df[self.plot_z]-self.df['Zero fit']

        #organize data into matrix for heatmap plot
        self.df_matrix = self.df.pivot_table(values=self.plot_z+' corrected',
                                             index=self.plot_y,
                                             columns=self.plot_x,
                                             aggfunc='first')
    
    #K-means cluster Z data
    def get_kmeans(self, n_clusters=2):
        kmeans = KMeans(n_clusters=n_clusters)
        data = np.array(self.df[self.plot_params['z']]).reshape(-1,1)
        k_fit = kmeans.fit(data)
        centers = k_fit.cluster_centers_.flatten()
        labels = k_fit.labels_
        lab0_min, lab1_min = 1e10, 1e10
        lab0_max, lab1_max = -1e10, -1e10
        for i, pt in enumerate(data):
            if labels[i] == 0:
                lab0_min = min([lab0_min, pt])
                lab0_max = max([lab0_max, pt])
            elif labels[i] == 1:
                lab1_min = min([lab1_min, pt])
                lab1_max = max([lab1_max, pt])
        #foreground/background cutoff limit
        cutoff = min([max([lab0_min, lab1_min]), min([lab0_max, lab1_max])])
        result = np.append(centers, [cutoff]) #include centers and cutoff
        return result

    #calculate volume    
    def get_volume(self, coord_vals, zero=0):

        data = np.array([[self.df_matrix.columns[coord[1]],
                                     self.df_matrix.index[coord[0]],
                                     self.df_matrix.iloc[coord[0], coord[1]]] for coord in coord_vals])

        data_x, data_y, data_z = data.T
        df_coord = pd.DataFrame({'x': data_x, 'y': data_y, 'z': data_z})
        df_coord_mat = df_coord.pivot_table(values='z',
                                            index='y', columns='x',
                                            aggfunc='first')
##        print(data_x,data_y,data_z)
##        print(data_x.shape,data_y.shape,data_z.shape)
##        df_shifted = df_matrix-zero
        df_coord_mat.fillna(zero, inplace=True)
##        print(np.trapz(df_coord_mat-zero,df_coord_mat.columns))

        try:
            #cubic interpolation of 2d data
            f_inter = interpolate.interp2d(df_coord_mat.columns,
                                           df_coord_mat.index,
                                           df_coord_mat, kind='cubic')
            num_interpol = 100
            x_inter = np.linspace(min(df_coord_mat.columns),
                                  max(df_coord_mat.columns),
                                  num_interpol)
            y_inter = np.linspace(min(df_coord_mat.index),
                                  max(df_coord_mat.index),
                                  num_interpol)
    ##        print(x_inter.shape,y_inter.shape)
    ##        z_inter = np.empty((num_interpol, num_interpol))
    ##        print(f_inter(x_inter[0,:], y_inter[0,:]).shape, x_inter[0,:].shape, y_inter[0,:].shape)
    ##        for i in range(num_interpol):
    ##            for j in range(num_interpol):
            z_inter = f_inter(x_inter, y_inter)
    ##        print(x_inter.shape,y_inter.shape, z_inter.shape)
    ####        print(z_inter, x_inter[:,0], y_inter[0,:])
            
            
            vol = np.trapz(np.trapz(df_coord_mat-zero,
                                    df_coord_mat.columns),
                           df_coord_mat.index)
    ##        print(vol)
            vol = np.trapz(np.trapz(z_inter-zero,
                                    x_inter),
                           y_inter)
        except Exception as e:
            vol=0
            print(e)
        return vol

    #get volume and contact angle of fitted cap
    #reference: https://mathworld.wolfram.com/SphericalCap.html
    def get_cap_prop(self, R, h):
##        h = z_max - zero #cap height
        vol = (1/3)*np.pi*((3*R)-h)*(h**2)
        angle = 90 - (np.arcsin((R-h)/R)*180/np.pi)
        return vol, angle


    def get_max_height(self, coord_vals, zero):
        data = np.array([[self.df_matrix.columns[coord[1]],
                                     self.df_matrix.index[coord[0]],
                                     self.df_matrix.iloc[coord[0], coord[1]]] for coord in coord_vals])

        data_x, data_y, data_z = data.T
        return data_z.max()-zero

##    def get_contact_radius(self, fit_out, zero):
##        return (fit_out['R']**2 - (zero-fit_out['c'])**2)**0.5

    def get_max_adhesion(self, jpk_anal, mode, coord_val): #find maximum adhesion within region
        plot_params =  jpk_anal.anal_dict[mode]['plot_parameters']
        x = plot_params['x']
        y = plot_params['y']
        z = plot_params['z']
        #TODO: clean it, change to query, avoid pivot
        df_adh =  jpk_anal.df[mode].pivot_table(values=z, index=y, columns=x,
                                                aggfunc='first')
        return max([df_adh.iloc[coord[0], coord[1]] for coord in coord_val])
        
        
    
from skimage.filters import sobel
from skimage import segmentation
from skimage.color import label2rgb
from skimage.exposure import histogram
from skimage.measure import regionprops
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

class ImageAnalyze:
    def __init__(self, jpk_anal, mode):
##        mode = 'Adhesion'
        plot_params =  jpk_anal.anal_dict[mode]['plot_parameters']
        x = plot_params['x']
        y = plot_params['y']
        z = plot_params['z']
        self.im_df =  jpk_anal.df[mode].pivot_table(values=z, index=y, columns=x,
                                                      aggfunc='first')
        self.im_data =  self.im_df.to_numpy()
##        self.jpk_anal = jpk_anal
        
    def segment_image(self, bg, fg, output_path=None):        
        self.im_sobel = sobel(self.im_data)
        self.markers = np.zeros_like(self.im_data)
        #set background
        self.markers[np.logical_and(self.im_data > bg[0],
                                    self.im_data < bg[1])] = 1
        #set foreground
        self.markers[np.logical_and(self.im_data > fg[0],
                                    self.im_data < fg[1])] = 2
        self.im_segment = segmentation.watershed(self.im_sobel, self.markers)
        
        im_segment2 = ndi.binary_fill_holes(self.im_segment - 1)
        self.im_labelled, _ = ndi.label(im_segment2)
        self.masked = np.ma.masked_where(self.im_labelled == 0,
                                    self.im_labelled)
        
        self.fig = plt.figure('Segments')
        self.ax = self.fig.add_axes([0.10, 0.3, 0.8, 0.6])
        self.ax.imshow(self.im_data, cmap='afmhot')
        self.ax.imshow(self.masked, cmap='rainbow')

        self.bbox = {}
        self.coords = {}
        for region in regionprops(self.im_labelled):
##            # take regions with large enough areas
##            if region.area >= 100:
                # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            self.bbox[region.label] = [self.im_df.columns[minc],
                                       self.im_df.index[minr],
                                       self.im_df.columns[maxc-1],
                                       self.im_df.index[maxr-1]]
            self.coords[region.label] = region.coords
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='white', linewidth=1)
            self.ax.add_patch(rect)
            self.ax.text(minc,minr,str(region.label),color='white',fontsize=12)
        
        self.ax.invert_yaxis()
        self.ax.grid(False)

        #create sliders for tweaking fg/bg params
        axis_bg = self.fig.add_axes([0.10, 0.1, 0.8, 0.03], facecolor='lightgoldenrodyellow')        
        self.slider_bg = Slider(axis_bg, 'BG', self.im_data.min(),
                                self.im_data.max(), valinit=bg[1], valfmt='%.1e')
        axis_fg = self.fig.add_axes([0.10, 0.15, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        self.slider_fg = Slider(axis_fg, 'FG', self.im_data.min(),
                                self.im_data.max(), valinit=fg[0], valfmt='%.1e')
        self.slider_bg.on_changed(self.update_segments)
        self.slider_fg.on_changed(self.update_segments)

        plt.show(block=True)
        self.fig.savefig(f'{output_path}/Segments.png', bbox_inches = 'tight',
                    transparent = True)
        
    #update sliders for segmentation
    def update_segments(self, val):
        #self.im_sobel = sobel(self.im_data)
        self.markers = np.zeros_like(self.im_data)
        #set background
        self.markers[np.logical_and(self.im_data > -1e10,
                                    self.im_data < self.slider_bg.val)] = 1
        #set foreground
        self.markers[np.logical_and(self.im_data > self.slider_fg.val,
                                    self.im_data < 1e10)] = 2
        self.im_segment = segmentation.watershed(self.im_sobel, self.markers)
        
        im_segment2 = ndi.binary_fill_holes(self.im_segment - 1)
        self.im_labelled, _ = ndi.label(im_segment2)
        self.masked = np.ma.masked_where(self.im_labelled == 0,
                                    self.im_labelled)
        
        self.ax.cla()
        self.ax.imshow(self.im_data, cmap='afmhot')
        self.ax.imshow(self.masked, cmap='rainbow')

        self.bbox = {}
        self.coords = {}
        for region in regionprops(self.im_labelled):
##            # take regions with large enough areas
##            if region.area >= 100:
                # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            self.bbox[region.label] = [self.im_df.columns[minc],
                                       self.im_df.index[minr],
                                       self.im_df.columns[maxc-1],
                                       self.im_df.index[maxr-1]]
            self.coords[region.label] = region.coords
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='white', linewidth=1)
            self.ax.add_patch(rect)
            self.ax.text(minc,minr,str(region.label),color='white',fontsize=12)
        
        self.ax.invert_yaxis()
        self.ax.grid(False)
        self.fig.canvas.draw_idle()

    def show_histogram(self):
        hist, hist_centers = histogram(self.im_data)
        fig = plt.figure('Histogram')
        ax = fig.add_subplot(111)
        ax.plot(hist_centers, hist, lw=2)
        plt.show(block=False)

##    def check_plot(self):
##        mode = 'Snap-in distance'
##        df = self.jpk_anal.df[mode]
##        plot_params =  self.jpk_anal.anal_dict[mode]['plot_parameters']
##        x = plot_params['x']
##        y = plot_params['y']
##        z = plot_params['z']        
##        df_data =  self.jpk_anal.df[mode].pivot_table(values=z, index=y, columns=x,
##                                                      aggfunc='first')
##        for key, val in self.bbox.items():
##            df_data_filter = df_data.iloc[val[0]:val[2], val[1]:val[3]]
##            fig2d = plt.figure(f'label {key}')
##            ax2d = fig2d.add_subplot(111)
##            im2d = ax2d.pcolormesh(df_data_filter.columns, df_data_filter.index,
##                                   df_data_filter, cmap='afmhot')
##            plt.show(block=False)

##        for key, val in self.coords.items():
##            data = np.array([df_data.columns[coord[0]],
##                             df_data.index[coord[1]],
##                             df_data.iloc[*coord]] for coord in val])
        
