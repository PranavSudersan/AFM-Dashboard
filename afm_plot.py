import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
import copy
import numpy as np
#import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.graph_objects as go
import scipy.ndimage as ndimage
#from afm_analyze import JPKAnalyze
#from plotly_viewer import PlotlyViewer
from plot2widget import PlotWidget

class AFMPlot:
    
    def __init__(self, jpk_anal=None, output_path=None):
        
        self.plotwin = None
        #mapping plot types (from ANALYSIS_MODE_DICT) to functions
        PLOT_DICT = {'2d': self.plot_2d,
                     '3d': self.plot_3d,
                     'line': self.plot_line}
        self.CLICK_STATUS = False #True when clicked first and plot doesn't exist

        if jpk_anal != None:
            self.jpk_anal = jpk_anal
            self.file_path = jpk_anal.file_path
            #plot data
            for mode in jpk_anal.df.keys():
                plot_params =  jpk_anal.anal_dict[mode]['plot_parameters']
                for plot_type in plot_params['type']:
                    fig = PLOT_DICT[plot_type](jpk_anal.df[mode], plot_params,
                                         file_path=output_path,
                                         points=plot_params['points_flag'],
                                        return_fig=True)
                    jpk_anal.anal_dict['Misc']['figure_list'].append(fig)

            #plt.show(block=False)
##        plt.pause(0.05)
##        if self.plotwin != None:
####            self.plotwin.show()
##            self.plotwin.app.exec_()
    
    def plot_2d(self, df, plot_params, file_path=None, points=False,return_fig=False):
        x = plot_params['x']
        y = plot_params['y']
        z = plot_params['z']
        title = plot_params['title']
        self.points_data = np.empty((0, 3), float)
        #organize data into matrix for heatmap plot
        df_data = df.pivot_table(values=z, index=y, columns=x,
                                 aggfunc='first')
        
        #plot
        plt.rcParams.update(plt.rcParamsDefault)
        fig2d = plt.figure(f'{title}')
        ax2d = fig2d.add_subplot(111)
        im2d = ax2d.pcolormesh(df_data.columns, df_data.index,
                               df_data, cmap='afmhot')
        ax2d.ticklabel_format(style='sci', scilimits=(0,0))
        ax2d.set_xlabel(x)
        ax2d.set_ylabel(y)
        fig2d.colorbar(im2d, ax=ax2d, label=z, format='%.1e')
        fig2d.suptitle(title)

        fig2d.savefig(f'{file_path}/{title}.png', bbox_inches = 'tight',
                      transparent = True)

        canvas = fig2d.canvas

##        rgb_data = np.fromstring(canvas.tostring_rgb(),
##                                 dtype=np.uint8, sep='')
##        rgb_data = rgb_data.reshape(canvas.get_width_height()[::-1] + (3,))

        if points == False:
            df_reference = df.pivot_table(values='Segment folder', index=y,
                                          columns=x, aggfunc='first')
            self.cid = fig2d.canvas.mpl_connect('button_press_event',
                                           lambda event: self.on_click(event,
                                                                       df_reference))
        else:
            self.cid = fig2d.canvas.mpl_connect('button_press_event',
                                           lambda event: self.on_click(event,
                                                                       df_data, points))
                
        #BUG: program doesn't end after callbacks
        fig2d.canvas.mpl_connect('close_event',
                                 lambda event: self.on_figclose(event, fig2d))
        plt.show(block=points)
        
        if return_fig == True:
            return fig2d


    def plot_3d(self, df, plot_params, file_path=None, points=False, return_fig=False):
        x = plot_params['x']
        y = plot_params['y']
        z = plot_params['z']
        #organize data into matrix for 3d plot
        df_data = df.pivot_table(values=z, index=y, columns=x,
                                 aggfunc='first')
        df_reference = df.pivot_table(values='Segment folder', index=y,
                                      columns=x, aggfunc='first')
        
        #plot
        fig = go.Figure(data=[go.Surface(z=df_data,
                                         x=df_data.columns,
                                         y=df_data.index)])
        fig.update_layout(title=plot_params['title'],
                          scene = dict(xaxis_title=x,
                                       yaxis_title=y,
                                       zaxis_title=z),
                          autosize=True)
        
        if return_fig == True:
            return fig
        #self.plotwin  = PlotlyViewer(fig)

        
##    def plot_3d(self, df, plot_params):
##        x = plot_params['x']
##        y = plot_params['y']
##        z = plot_params['z']
##
##        self.fig3d = plt.figure('3D Surface')
##        self.ax3d = self.fig3d.gca(projection='3d')
##        surf = self.ax3d.plot_trisurf(df[x], df[y], df[z])#, cmap='afmhot')
##        self.ax3d.ticklabel_format(style='sci', scilimits=(0,0))
##        self.ax3d.set_xlabel(x)
##        self.ax3d.set_ylabel(y)
##        self.ax3d.set_zlabel(z)
##        self.fig3d.suptitle(plot_params['title'])
##        ##fig3d.colorbar(surf, shrink=0.5, aspect=5)

    def plot_line(self, df, plot_params, label_text=None, file_path=None, points=False,
                  color=None, return_fig=False):
        x = plot_params['x']
        y = plot_params['y']
        style = plot_params['style']
        
        if self.CLICK_STATUS == False:
            self.init_fd_plot()
            self.CLICK_STATUS = True
            self.cursor_index = [0, -1]
        #functionality to make extend/retract colors same or different
        if color in df.columns:
            hue_param = color
            color_param = None
        elif color == None:
            hue_param = None
            color_param = None
        else:
            hue_param = None
            color_param = color
            
        sns.lineplot(x=x, y=y, style=style,hue=hue_param,
                     data=df, ax=self.ax_fd,
                     label = label_text, sort=False, ci=None,
                     color=color_param)
##        self.ax_fd.plot(df[x], df[y])
        self.ax_fd.ticklabel_format(style='sci', scilimits=(0,0))
        self.ax_fd.set_xlabel(x)
        self.ax_fd.set_ylabel(y)
        self.fig_fd.suptitle(plot_params['title'])        
        
        if return_fig == True:
            return self.fig_fd
        
        #plt.show(block=False)
##        df.to_excel('test-fd-data.xlsx')

    def init_fd_plot(self): #initialize force-distance plot
        sns.set_theme(palette = 'husl')
        #self.sourceLabel = None
        #self.fig_fd = plt.figure('Line plot')
        self.fig_fd = Figure(figsize=(11, 5), dpi=100)
        self.ax_fd = self.fig_fd.add_subplot(111)
        self.fd_fit_line = self.ax_fd.plot([], [], 'k:') #fd fit line object
        #print('before')
##        self.fig_fd.canvas.mpl_connect('close_event',
##                                       lambda event: self.on_close(event))
        self.plotWidget = PlotWidget(fig = self.fig_fd,
                                         cursor1_init=0,
                                         cursor2_init=1,
                                         fixYLimits = False,
                                         method = self.updatePosition)
        self.plotWidget.wid.clicked_artist = None

    def updatePosition(self, trigger=False):

        # final_pos = tuple(self.plotWidget.wid.axes.transLimits.transform
        #                   ((self.plotWidget.wid.final_pos)))
##        if self.plotWidget.wid.clicked_artist == self.fitTextBox:
##            self.fit_pos = list(self.fitTextBox.get_position())
        # elif self.plotWidget.wid.clicked_artist == self.legend_main:
        #     pos = str(tuple(self.legend_main.get_window_extent()))
        #     self.configPlotWindow.plotDict['plot settings']['legend position'].setText(pos)
        if trigger == False:
            if self.plotWidget.wid.clicked_artist in [self.plotWidget.wid.cursor1,
                                                      self.plotWidget.wid.cursor2]:
                condition = True
            else:
                condition = False
        else:
            condition = True
        
        if condition == True:
##            if self.sourceLabel != None:
            xdata =  self.xAxisData
            ydata = self.yAxisData
            #print(xdata,ydata)
            x1 = self.plotWidget.wid.cursor1.get_xdata()[0]                                
            #x1_ind = np.searchsorted(xdata, x1)
            x1_ind = (np.abs(xdata-x1)).argmin()
            #print(xdata,x1,x1_ind,np.searchsorted(xdata, [x1]))
##            if len(self.sourceLabel.text().split(',')) == 2:
            if self.plotWidget.wid.cursor2 != None:
                x2 = self.plotWidget.wid.cursor2.get_xdata()[0]
                #x2_ind = np.searchsorted(xdata, [x2])[0]
                x2_ind = (np.abs(xdata-x2)).argmin()
                xstart = min(x1_ind, x2_ind)
                xend = max(x1_ind, x2_ind)
                xend = xend-1 if xend == len(xdata) else xend            
##                    self.sourceLabel.setText(str(xstart) + ',' + str(xend))
            else:
                xstart = x1_ind-1 if x1_ind == len(xdata) else x1_ind
                xend = None
##                    self.sourceLabel.setText(str(xstart))
                # self.sourceLabel = None
            self.cursor_index = [xstart, xend]
            #print('cursors', xstart, xend)
            #print(x1,x1_ind,x2,x2_ind)
            #fitting
            fit_slice = slice(xstart,xend)
            #print("Fit slice", fit_slice)
            retract_fit = np.polyfit(xdata[fit_slice],
                                     ydata[fit_slice],self.fd_fit_order) #CHECK FIT ORDER
            fit_poly = np.poly1d(retract_fit)
            self.ax_fd.autoscale(enable=False)
            self.fd_fit_line[0].set_xdata(xdata)
            self.fd_fit_line[0].set_ydata(fit_poly(xdata))
            
#             x0 = min(xdata)
#             dydx = 2*retract_fit[0]*x0 + retract_fit[1]
#             y0 = fit_poly(x0)
            
#             force_sobel = ndimage.sobel(ydata) #sobel transform
#             idx_min = np.argmin(force_sobel)
#             snapin_distance = xdata[idx_min] - x0
            
#             intercept = (abs(y0-ydata[0])/dydx)
            
#             print(y0-ydata[0], dydx, intercept, snapin_distance, intercept/snapin_distance)
#             #print('polyfit x intercept',min(np.roots(retract_fit))-min(xdata))
#             print('cursor range', abs(xdata[xstart]-xdata[xend]))

    
    def on_close(self, event):
        self.CLICK_STATUS = False

    def on_figclose(self, event, fig):
        fig.canvas.mpl_disconnect(self.cid)
##        self.jpk_anal.data_zip.close() #CHECK THIS
        
    def on_click(self, event, df_ref, points=False):
        #print('click')
        x, y = event.xdata, event.ydata
        if x != None and y != None:
            #get segment path corresponding to clicked position
            col = sorted([[abs(a - x), a] for a in df_ref.columns],
                         key=lambda l:l[0])[0][1]
            ind = sorted([[abs(a - y), a] for a in df_ref.index],
                         key=lambda l:l[0])[0][1]
            

            if points == False:
                segment_path = df_ref[col][ind]
                print(segment_path, col, ind)
                #TODO: clean and organize this up
                mode = 'Force-distance'
                fd_data = self.jpk_anal
                segment_path_old = fd_data.segment_path
    ##            anal_dict_old = fd_data.anal_dict.copy()
                fd_data.clear_output(mode) #clear previous data
                fd_data.segment_path = segment_path
    ##            fd_data.anal_dict = fd_data.ANALYSIS_MODE_DICT[mode].copy()
    ##            print('old')
    ##            print(mode_dict_old)
                fd_data.get_data([mode])
                
                #fd_data = JPKAnalyze(self.file_path, mode, segment_path)
                plot_params = fd_data.ANALYSIS_MODE_DICT[mode]['plot_parameters']

                label_text = f'x={"{:.2e}".format(col)}, y={"{:.2e}".format(ind)}'
                self.plot_line(fd_data.df[mode], plot_params, label_text=label_text)

                #legend remove duplicates
                handles, labels = self.ax_fd.get_legend_handles_labels()            
                leg_dict = dict(zip(labels[::-1],handles[::-1]))
                self.ax_fd.get_legend().remove()
                leg = self.ax_fd.legend(leg_dict.values(), leg_dict.keys())
                leg.set_draggable(True, use_blit=True)

                #CHECK
                fd_data.segment_path = None
    ##            fd_data.anal_dict = anal_dict_old

                self.fig_fd.show()
            else:
                z_value = df_ref[col][ind]
                self.points_data = np.append(self.points_data,
                                             np.array([[col,ind,z_value]]),
                                             axis=0)
                #print(col, ind, z_value)
            

    def plot_2dfit(self, fit_data, df_raw, plot_params, file_path=None):
        x = plot_params['x']
        y = plot_params['y']
        z = plot_params['z']
        z_raw = f'{z}_raw'
        z_fit = f'{z}_fit'

##        color_limits = [df[z_raw].min(), df[z_raw].max()]
##        title_text = ', '.join([f'{k}' + '={:.1e}'.format(v) \
##                                for k, v in fit_output.items()])
        title_text = 'Fitting result'

        #data reshape
##        df_raw = df.pivot_table(values=z_raw, index=y, columns=x,
##                                       aggfunc='first')
##        df_filtered = df.query(f'{z_fit}>={color_limits[0]}')
##        df_fit = fit_data.pivot_table(values=z_fit, index=y,
##                                           columns=x, aggfunc='first')

        #plot
        fig = go.Figure()
        fig.add_trace(go.Surface(legendgroup="Raw",
                                 name='Raw',
                                 z=df_raw,
                                 x=df_raw.columns,
                                 y=df_raw.index,
                                 opacity=0.7,
                                 colorscale ='Greens',
                                 reversescale=True,
                                 showlegend=True,
                                 showscale=False))
        #i = 0
        for key, val in fit_data.items():
##            print(key)
            #leg = True if i==0 else False
            fig.add_trace(go.Surface(#legendgroup="Fit",
                                     name=f'Fit-{key}',
                                     z=val[z],#df_fit,
                                     x=val[x],#df_fit.columns,
                                     y=val[y],#df_fit.index,
                                     opacity=0.7,
                                     colorscale ='Reds',
                                     reversescale=True,
                                     showlegend=True,#leg
                                     showscale=False))
            #i = 1
##        fig.update_traces(contours_z=dict(show=True, usecolormap=False,
##                                          highlightcolor="limegreen",
##                                          project_z=True))
        z_range = [df_raw.min().min(),1.2*df_raw.max().max()]
##        print(z_range)
        fig.update_layout(title=title_text,
                          scene = dict(xaxis_title=x,
                                       yaxis_title=y,
                                       zaxis_title=z,
                                       xaxis=dict(ticksuffix='m'),
                                       yaxis=dict(ticksuffix='m'),                                       
                                       zaxis=dict(range=z_range,
                                                  ticksuffix='m'),
                                       aspectratio={"x": 1, "y": 1, "z": 0.4}
                                       ),
                          autosize=True)

        fig.write_html(f'{file_path}/3d_jumpin_distance.html')
        
        #self.plotwin  = PlotlyViewer(fig)
        
        return fig

def simul_plot1(simu_df):
    sns.set_context("talk")
    sns.set_style("ticks")
##    fig = plt.figure('Simulation data')
##    
##    ax1 = fig.add_subplot(1,1,1)
    mk_num = len(simu_df['Top_Angle'].unique())
    g = sns.lmplot(x='Contact_Radius',y='Force_Calc',hue='Top_Angle',
                 #style='Top_Angle',
               data=simu_df,
                 #markers=['o']*mk_num,dashes=False,
                 legend='full',palette='flare',
               #ax=ax1,
               order=5, ci=None,
                   height=8, aspect=1.3)
    ax1 = g.ax
    ax1.axhline(y=0, color='0.8', dashes=(1, 1), zorder=0)
    
    ax1.set_title('Simulation data: Adhesion force')
    ax1.set_xlabel('Drop size, R/s')
    ax1.set_ylabel(r'$F/2\pi \gamma s$')
    leg = g.legend
    #leg = ax1.get_legend()
##    leg.remove()
    
##    ax2 = fig.add_subplot(1,2,2)
##    mk_num = len(simu_df['Top_Angle'].unique())
##    sns.lineplot(x='Contact_Radius',
##                 y='Average Wetted Height',hue='Top_Angle',
##                 style='Top_Angle',data=simu_df,
##                 markers=['o']*mk_num,dashes=False,
##                 legend='full',palette='flare', ax=ax2)
##
##    ax2.set_title('Wetted length')
##    ax2.set_xlabel('Drop size, R/s')
##    ax2.set_ylabel('w/s')
##    leg = ax2.get_legend()
    leg.set_title('Contact angle')
    fig = g.fig
    #plt.show(block=True)

    return fig


def simul_plot2(simu_df):
    sns.set_context("talk")
    sns.set_style("ticks")
    Rs = simu_df['Contact_Radius'].iloc[0]
##    fig = plt.figure(f'Simulation data FD Rs={Rs}')
##    
##    ax1 = fig.add_subplot(1,1,1)
    mk_num = len(simu_df['Top_Angle'].unique())
    g = sns.lmplot(x='Height',y='Force_Calc',hue='Top_Angle',
                 #style='Top_Angle',
               data=simu_df,
                 #markers=['o']*mk_num,dashes=False,
                 legend='full',palette='flare',
               #ax=ax1,
               order=2, ci=None,
                   height=8, aspect=1.3)
    ax1 = g.ax
    ax1.axhline(y=0, color='0.8', dashes=(1, 1), zorder=0)
    
    ax1.set_title(f'Simulation data (FD): R/s={Rs}')
    ax1.set_xlabel('Height, h/s')
    ax1.set_ylabel(r'$F/2\pi \gamma s$')
    #leg = ax1.get_legend()
    leg = g.legend
##    leg.remove()
##    
##    ax2 = fig.add_subplot(1,2,2)
##    mk_num = len(simu_df['Top_Angle'].unique())
##    sns.lineplot(x='Contact_Radius',
##                 y='Average Wetted Height',hue='Top_Angle',
##                 style='Top_Angle',data=simu_df,
##                 markers=['o']*mk_num,dashes=False,
##                 legend='full',palette='flare', ax=ax2)
##
##    ax2.set_title('Wetted length')
##    ax2.set_xlabel('Drop size, R/s')
##    ax2.set_ylabel('w/s')
##    leg = ax2.get_legend()
    leg.set_title('Contact angle')
    fig = g.fig
    #plt.show(block=True)

    return fig


def simul_plot3(simu_df):
    sns.set_context("talk")
    sns.set_style("ticks")
##    fig = plt.figure('Simulated contact angle')
##    
##    ax1 = fig.add_subplot(1,1,1)
    mk_num = len(simu_df['Contact_Radius'].unique())
    g = sns.lmplot(x='Rupture_Distance',y='Top_Angle',hue='Contact_Radius',
                 #style='Contact_Radius',
                 data=simu_df,
                 #markers=['o']*mk_num,dashes=False,sort=False,
                 legend='full',palette='flare',
                 #ax=ax1,
                 order=3, ci=None)
    ax1 = g.ax
    ax1.set_title('Simulation data: rupture distance')
    ax1.set_xlabel('Rupture distance, r/s')
    ax1.set_ylabel('Contact angle')
    #leg = ax1.get_legend()
    leg = g.legend

    leg.set_title('Drop size, R/s')
    fig = g.fig
    #plt.show(block=True)

    return fig


##    def plot_2dfit(self, df, plot_params, fit_output):
##        x = plot_params['x']
##        y = plot_params['y']
##        z = plot_params['z']
##        z_raw = f'{z}_raw'
##        z_fit = f'{z}_fit'
##
##        color_limits = [df[z_raw].min(), df[z_raw].max()]
##
##        #fit data reshape
##        df_fit = df.pivot_table(values=z_fit, index=y, columns=x,
##                                aggfunc='first')
##        #plot
##        fig = plt.figure('Fit: 2D map')
##        ax_fit = fig.add_subplot(111)
##        im_fit = ax_fit.pcolormesh(df_fit.columns, df_fit.index,
##                                   df_fit, cmap='afmhot',
##                                   vmin=color_limits[0],
##                                   vmax=color_limits[1])
##        ax_fit.ticklabel_format(style='sci', scilimits=(0,0))
##        ax_fit.set_xlabel(x)
##        ax_fit.set_ylabel(y)
##        title_text = ', '.join([f'{k}' + '={:.1e}'.format(v) \
##                                for k, v in fit_output.items()])
##        ax_fit.set_title(title_text)
##        fig.colorbar(im_fit, ax=ax_fit, label=z)
##        
##        fig.tight_layout()
##
##        #3d plot
##        z_lim = self.ax3d.get_zlim3d() #z limits of raw 3D
##        df_filtered = df.query(f'{z_fit}>={z_lim[0]}')
##        fig3d = plt.figure('Fit: 3D')
##        ax3d = fig3d.gca(projection='3d')
##        surf = ax3d.plot_trisurf(df_filtered[x],df_filtered[y],
##                                 df_filtered[z_fit])#, cmap='afmhot')
##        ax3d.ticklabel_format(style='sci', scilimits=(0,0))
##        ax3d.set_xlabel(x)
##        ax3d.set_ylabel(y)
##        ax3d.set_zlabel(z)
##        fig3d.suptitle(title_text)        
##        ax3d.set_zlim3d(color_limits)
##        ax3d.set_xlim((df[x].min(), df[x].max()))
##        ax3d.set_ylim(df[y].min(), df[y].max())
##        
##        plt.show(block=False)
##    
