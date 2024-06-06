import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
import PIL
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import io, base64

#plot theme dictionary
THEME_DICT={'dark': {'plotly':'plotly_dark',
                     'matplotlib':'dark_background',
                     'fontcolor': 'white',
                     'bgcolor': 'black',
                     'axiscolor': ['OrangeRed', 'Cyan', 'LimeGreen','Yellow']
                    },
            'light':{'plotly':'plotly_white',
                     'matplotlib':'default',
                     'fontcolor': 'black',
                     'bgcolor': 'white',
                     'axiscolor': ['Red', 'Blue', 'Green', 'Darkorange']
                    },
           }

def set_theme(theme):
    global THEME
    THEME = theme

# def set_theme(theme):
#     THEME = 
#convert matplotlib colormaps to plotly format
def matplotlib_to_plotly(cmap_name, num=255):
    cmap = matplotlib.cm.get_cmap(cmap_name)
    h = 1.0/(num-1)
    pl_colorscale = []

    for k in range(num):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale

#initialize default colourmap "afmhot"
cm_afmhot = matplotlib_to_plotly('afmhot', 255)

# create plot with multiple secondary y axes, grouped by data columns specified in line_group, symbol and line_dash   
# secondary y axis created based on values populated in multiy_col column of data. yvars list can be used to limit the number of
#secondary y axis created. x, y, line_group, symbol, hover_name and line_dash are passed to px.line (check plotly documentation for that)
#data must be in "long form", i.e. names to be made into y axis must be inside "multiy_col" column and its corresponding values
#must be in "y" column. "x" column must contain the common data to be plotted in x axis. Use pandas.melt to convert to long form.
def plotly_multiyplot(data, multiy_col, yvars, x, y, fig=None, yax_dict=None,
                      line_group=None, symbol=None, color=None, line_dash=None, hover_name=None, font_dict=None):
    # color_list = ['magenta', 'yellow', 'lime', 'cyan']
    # color_list = ['OrangeRed','Yellow','LimeGreen','Cyan']
    color_list = THEME_DICT[THEME]['axiscolor']
    # if font_dict == None:
    #     font_dict=dict(family='Arial', size=22, color=THEME_DICT[THEME]['fontcolor'])#color='white')
    if font_dict == None:
        font_dict=dict(family='Arial',size=22)
    font_dict['color']=THEME_DICT[THEME]['fontcolor']
    if yax_dict == None:
        yax_dict = {}

    if fig == None:
        # fig = go.FigureWidget()
        # fig.update_layout(font=font_dict,  # font formatting
        #                   template=THEME_DICT[THEME]['plotly'],#'plotly_dark',
        #                   plot_bgcolor=THEME_DICT[THEME]['bgcolor'],#'black',  # background color
        #                   height=500, width=1100, title_text="",
        #                   margin=dict(t=50, b=0, l=0, r=0),
        #                   showlegend=False)
        fig = plotly_multiyplot_initax(fig=None, yvars=yvars, 
                                       yax_dict=yax_dict, font_dict=font_dict)
        
    yvars_old = list(yax_dict.keys())

    if all(var in yvars for var in yvars_old) == False:
        yax_dict.clear()
        yvars_new = yvars
        i = 0
    elif list(np.sort(yvars)) == list(np.sort(yvars_old)):
        fig.data = []
        yvars_new = yvars_old #yvars
        i = 0
    else:
        i = len(yvars_old)
        yvars_new = [yvar_i for yvar_i in yvars if yvar_i not in yvars_old]
        # yvars = yvars_old + yvars_new

    for yvars_i in yvars_new:
        data_i = data[data[multiy_col]==yvars_i]
        trace_i = px.line(data_i, x=x, y=y, line_group=line_group, symbol=symbol, color=color,
                          hover_name=hover_name, line_dash=line_dash, symbol_sequence = ['circle'])
        if color == None:
            trace_i.update_traces(yaxis=f'y{i+1}', line_color=color_list[i],# if color==None else None,
                                  marker=dict(size=1))
        else:
            trace_i.update_traces(yaxis=f'y{i+1}', #line_color=color_list[i] if color==None else None,
                                  marker=dict(size=1))
        for trace_data_i in trace_i.data:
            fig.add_trace(trace_data_i)
        yax_dict[yvars_i] = f'y{i+1}'
        i += 1
    
    fig.update_xaxes(domain=[0.15, 0.85], title_text='Z',
                     showline=True,
                     showgrid=False,
                     zeroline=False,
                     linecolor=THEME_DICT[THEME]['fontcolor'],#'white',
                     linewidth=2.4,
                     ticks='outside',
                     tickfont=font_dict,
                     title_font=font_dict,
                     tickwidth=2.4,
                     tickcolor=THEME_DICT[THEME]['fontcolor']#'white'
                    )
    
    return fig, yax_dict

# delete and recreate secondary y axes based on a given yvars and initialise the plot. yax_dict is a dictionary of previously made
# secondary y axis in the plot, which is compare to create new y axes or recreate everything.
def plotly_multiyplot_initax(fig, yvars, yax_dict, font_dict=None, height=500, width=1100, 
                             margin=dict(t=50, b=0, l=0, r=0)):
    if fig == None:
        fig = go.FigureWidget()
        fig.update_layout(font=font_dict,  # font formatting
                          template=THEME_DICT[THEME]['plotly'],#'plotly_dark',
                          plot_bgcolor=THEME_DICT[THEME]['bgcolor'],#'black',  # background color
                          height=height, width=width, title_text="",
                          margin=margin,
                          showlegend=False)
    # color_list = ['magenta', 'yellow', 'lime', 'cyan']
    # color_list = ['OrangeRed','Yellow','LimeGreen','Cyan']
    color_list = THEME_DICT[THEME]['axiscolor']
    # color_list = ['#FF4500','#00FFFF','#32CD32','#FFFF00']
    # if font_dict == None:
    #     font_dict=dict(family='Arial',size=22,color=THEME_DICT[THEME]['fontcolor'])#color='white')
    if font_dict == None:
        font_dict=dict(family='Arial',size=22)
    font_dict['color']=THEME_DICT[THEME]['fontcolor']
    yvars_old = list(yax_dict.keys())
    if all(var in yvars for var in yvars_old) == False:
        yvars_new = yvars
        i = 0
        try:
            fig.layout.yaxis = {}
            fig.layout.yaxis2 = {}
            fig.layout.yaxis3 = {}
            fig.layout.yaxis4 = {}
            fig.data = []
        except:
            print('too many axis')
    else:
        i = len(yvars_old)
        yvars_new = [yvar_i for yvar_i in yvars if yvar_i not in yvars_old]
    
    for yvars_i in yvars_new:
        if i == 0:
            fig.update_layout(yaxis=dict(
                title_text=yvars_i,                
                showline=True,
                showgrid=False,
                zeroline=False,
                linecolor=color_list[i],
                linewidth=2.4,                
                ticks='outside',
                titlefont=dict(color=color_list[i], size=font_dict['size'], family=font_dict['family']),
                tickfont=dict(color=color_list[i])
            ))
        elif i == 1:
            fig.update_layout({'yaxis2':dict(
                title_text=yvars_i,                
                showline=True,
                showgrid=False,
                zeroline=False,
                linecolor=color_list[i],
                linewidth=2.4,                
                ticks='outside',
                titlefont=dict(color=color_list[i], size=font_dict['size'], family=font_dict['family']),
                tickfont=dict(color=color_list[i]),
                overlaying='y',
                anchor="x",
                side='right'
            )})
        else:
            fig.update_layout({f'yaxis{i+1}': dict(
                title_text=yvars_i,
                showline=True,
                showgrid=False,
                zeroline=False,
                #showticklabels=True,
                linecolor=color_list[i],
                linewidth=2.4,
                ticks='outside',
                titlefont=dict(color=color_list[i], size=font_dict['size'], family=font_dict['family']),
                tickfont=dict(color=color_list[i]),
                overlaying='y',
                anchor="free",
                side='right' if i % 2 == 1 else 'left',
                position=1 if i % 2 == 1 else 0
            )})
            
        i += 1
    return fig
            
#plotly version of seaborn's pairplot to plot relational xy data in a grid for all cols in data. Diagonal of the gird shows histogram, use
#nbins to adjust histogram bins
def plotly_pairplot(data, fig=None, cols=None, hue=None, diag_kind='hist', nbins = 50, font_dict=None):
    # if font_dict == None:
    #     font_dict=dict(family='Arial', size=22, color=THEME_DICT[THEME]['fontcolor'])#color='white')
    if font_dict == None:
        font_dict=dict(family='Arial',size=22)
    font_dict['color']=THEME_DICT[THEME]['fontcolor']
    # font_dict=dict(family='Arial',
    #            size=22,
    #            color='black'
    #        )
    if cols is None:
        cols = data.columns
    num_cols = len(cols)
    
    if fig == None:
        fig = plotly_pairplot_initax(fig, num_cols, font_dict=font_dict)
        
    fig.data = [] #clear old data

    # Assign colors to unique categories in the hue column
    if hue is not None:
        unique_categories = data[hue].unique()
        colors = px.colors.qualitative.Plotly[:len(unique_categories)]
        color_map = dict(zip(unique_categories, colors))
    else:
        color_map = None
    
    for i, var1 in enumerate(cols):
        for j, var2 in enumerate(cols):
            if i == j:
                if diag_kind == 'hist':
                    for name, group in data.groupby(hue):
                        if name != hue:
                            color = color_map[name] if color_map is not None else None
                            if group[var1].isna().any(): #dummy data for histogram if channel data doesn't exist
                                d_len = len(group[var1])
                                group[var1] = np.linspace(0,1, d_len)
                            plot_dummy = go.Scatter(x=group[var1], y=group[var1], yaxis=f'y{((num_cols+2)*i)+1}', visible=False)
                            fig.add_trace(plot_dummy, row=i+1, col=j+1)
                            plot_hist = go.Histogram(x=group[var1], marker=dict(color=color), nbinsx=nbins, name=name, opacity=0.8,
                                                     showlegend=True if i==0 else False)

                            fig.add_trace(plot_hist, row=i+1, col=j+1, secondary_y=True)

                    fig.update_layout({f'yaxis{(num_cols+2)*i+2}': dict(showgrid=False,     # Hide grid lines
                                                        zeroline=False,     # Hide zero line
                                                        showline=False,     # Hide axis line
                                                        ticks='',           # Hide ticks
                                                        showticklabels=False, # Hide tick labels
                                                        title=''
                                                    )})
                elif diag_kind == 'kde':
                    # Plot KDE using histogram for simplicity
                    for name, group in data.groupby(hue):
                        if name != hue:
                            color = color_map[name] if color_map is not None else None
                            plot_hist = go.Histogram(x=group[var1], marker=dict(color=color), 
                                                     name=name, opacity=0.8, histnorm='probability density')
                            fig.add_trace(plot_hist, row=i+1, col=j+1)
                            if i == 0 and j == 0:  # Adjust y-axis range only for the top left histogram
                                fig.update_yaxes(range=[0, max(plot_hist['x']) * 1.1], row=i+1, col=j+1)
                    fig.update_yaxes(matches=None, row=i+1, col=j+1)
            else:
                for name, group in data.groupby(hue):
                    if name != hue:
                        color = color_map[name] if color_map is not None else None
                        unique_combinations = group.groupby(['segment', 'curve number'])
                        for (segment, curve_number), group2 in unique_combinations:
                            fig.add_trace(go.Scatter(x=group2[var2], y=group2[var1], mode='lines+markers',
                                                     marker=dict(color=color, size=5), name=name, showlegend=(i==0 and j==0)),
                                          row=i+1, col=j+1)
            if j == 0:
                fig.update_yaxes(title_text=var1, row=i+1, col=j+1,
                                tickfont=font_dict, secondary_y=False)
            if i == num_cols-1:
                fig.update_xaxes(title_text=var2, row=i+1, col=j+1,
                                tickfont=font_dict)
    
    return fig

#intialize grid layout of figure for pairplot with numxnum subplots. 
#Call this function if you want dynamically change the number of grid elements in an
#already created fig.
def plotly_pairplot_initax(fig, num, font_dict=None):
    # if font_dict == None:
    #     font_dict=dict(family='Arial', size=22, color=THEME_DICT[THEME]['fontcolor'])#color='white')
    if fig == None:
        fig = go.FigureWidget()
    if font_dict == None:
        font_dict=dict(family='Arial',size=22)
    font_dict['color']=THEME_DICT[THEME]['fontcolor']
    # font_dict=dict(family='Arial',size=22,color='black')
    specs = [[{} for _ in range(num)] for _ in range(num)]
    for i in range(num):
        specs[i][i] = {"secondary_y": True}
    
    fig.data = []
    fig.layout = {}
    
    sp.make_subplots(rows=num, cols=num, figure=fig, specs=specs,
                     shared_xaxes=True, shared_yaxes=True,
                     vertical_spacing=0.01, horizontal_spacing=0.01)    
    fig.update_layout(legend=dict(orientation="h",yanchor="top",y=-0.1,xanchor="center",x=0.5, font_size=18),
                  font=font_dict,  # font formatting
                  plot_bgcolor=THEME_DICT[THEME]['bgcolor'],#'black',  # background color
                      template=THEME_DICT[THEME]['plotly'],#'plotly_dark',
                  barmode='overlay', showlegend=True, 
                  width=1100, height=1100,
                  title='', #template='plotly_white',  #"plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"
                  margin=dict(t=50, b=0, l=0, r=0))  # Adjust margins to remove white space

    for i in range(num):
        for j in range(num):
            # x and y-axis formatting
            fig.update_yaxes(showline=True,  # add line at x=0
                             showgrid=False,
                             zeroline=False,
                             linecolor=THEME_DICT[THEME]['fontcolor'],#'white',  # line color
                             linewidth=2.4, # line size
                             ticks='outside',  # ticks outside axis
                             tickfont=font_dict, # tick label font
                             title_font=font_dict,
                             tickwidth=2.4,  # tick width
                             tickcolor=THEME_DICT[THEME]['fontcolor'],#'white',  # tick color
                             row=i+1, col=j+1, secondary_y=False)
            fig.update_xaxes(showline=True,
                             showgrid=False,
                             zeroline=False,
                             #showticklabels=True,
                             linecolor=THEME_DICT[THEME]['fontcolor'],#'white',
                             linewidth=2.4,
                             ticks='outside',
                             tickfont=font_dict,
                             title_font=font_dict,
                             tickwidth=2.4,
                             tickcolor=THEME_DICT[THEME]['fontcolor'],#'white',
                             row=i+1, col=j+1)
            if i==j:
                fig.update_layout({f'yaxis{(num+2)*i+2}': dict(showgrid=False,     # Hide grid lines
                                                        zeroline=False,     # Hide zero line
                                                        showline=False,     # Hide axis line
                                                        ticks='',           # Hide ticks
                                                        showticklabels=False, # Hide tick labels
                                                        title=''
                                                    )})
    return fig

#line plot x vs y grouped by color column of dataframe data.
def plotly_lineplot(data, x, y, color=None, line_group=None, line_dash=None, symbol=None, 
                    height=400, width=500, font_dict=None, color_discrete_sequence=None, line_dash_sequence=None,
                   symbol_sequence=None):    
    fig = px.line(data, x=x, y=y, color=color, line_group=line_group, line_dash=line_dash,
                  symbol=symbol, color_discrete_sequence=color_discrete_sequence, line_dash_sequence=line_dash_sequence,
                  symbol_sequence=symbol_sequence)
    # if font_dict == None:
    #     font_dict=dict(family='Arial',size=16,color=THEME_DICT[THEME]['fontcolor'])#'white')
    if font_dict == None:
        font_dict=dict(family='Arial',size=16)
    font_dict['color']=THEME_DICT[THEME]['fontcolor']
    fig.update_xaxes(showline=True,
                     showgrid=False,
                     zeroline=False,
                     linecolor=THEME_DICT[THEME]['fontcolor'],#'white',
                     linewidth=1,
                     ticks='outside',
                     tickfont=font_dict,
                     title_font=font_dict,
                     tickwidth=1,
                     tickcolor=THEME_DICT[THEME]['fontcolor']#'white'
                    )
    fig.update_yaxes(showline=True,
                     showgrid=False,
                     zeroline=False,
                     linecolor=THEME_DICT[THEME]['fontcolor'],#'white',
                     linewidth=1,
                     ticks='outside',
                     tickfont=font_dict,
                     title_font=font_dict,
                     tickwidth=1,
                     tickcolor=THEME_DICT[THEME]['fontcolor']#'white'
                    )
    fig.update_layout(font=font_dict,
                      template=THEME_DICT[THEME]['plotly'],#'plotly_dark',
                      autosize=False,
                      height=height, 
                      width=width,
                      legend=dict(font=font_dict,
                                  yanchor="top",
                                  y=0.99,
                                  xanchor="right",
                                  x=1.01),
                      plot_bgcolor=THEME_DICT[THEME]['bgcolor'],#"white",
                      margin=dict(t=0,l=0,b=0,r=0))
    return fig

#plot heat map. here x,y are 1d arrays and z is 2d matrix array
def plotly_heatmap(x=None, y=None, z_mat=None, color=cm_afmhot, style='full', height=400, width=400, font_dict=None):
    fig = go.Figure(data=go.Heatmap(z=z_mat, x=x, y=y, type = 'heatmap', colorscale =color, 
                                    zmin=np.percentile(z_mat,1, method='midpoint'),
                                    zmax=np.percentile(z_mat,99, method='midpoint')
                                   )
                   )
    # if font_dict == None:
    #     font_dict=dict(family='Arial',size=16,color=THEME_DICT[THEME]['fontcolor'])#'white')
    if font_dict == None:
        font_dict=dict(family='Arial',size=16)
    font_dict['color']=THEME_DICT[THEME]['fontcolor']
    
    if style == 'clean':
        fig.update_traces(showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(font=font_dict,
                          template=THEME_DICT[THEME]['plotly'],#'plotly_dark',
                          autosize=False,
                          height=height, 
                          width=width,
                          margin=dict(t=0,l=0,b=0,r=0))
    elif style == 'full':
        fig.update_traces(showscale=True)
        # font_dict=dict(family='Arial',size=16,color='black')
        fig.update_xaxes(showline=True,
                         linecolor=THEME_DICT[THEME]['fontcolor'],#'white',
                         linewidth=1,
                         ticks='outside',
                         mirror='ticks',
                         tickfont=font_dict,
                         title_font=font_dict,
                         tickwidth=1,
                         tickcolor=THEME_DICT[THEME]['fontcolor'],#'white',
                        showticklabels=True)
        fig.update_yaxes(showline=True,
                         linecolor=THEME_DICT[THEME]['fontcolor'],#'white',
                         linewidth=1,
                         ticks='outside',
                         mirror='ticks',
                         tickfont=font_dict,
                         title_font=font_dict,
                         tickwidth=1,
                         tickcolor=THEME_DICT[THEME]['fontcolor'],#'white',
                        showticklabels=True)
        fig.update_layout(font=font_dict,
                          template=THEME_DICT[THEME]['plotly'],#'plotly_dark',
                          autosize=False,
                          height=400, 
                          width=1.2*400,
                          plot_bgcolor=THEME_DICT[THEME]['bgcolor'],#"white",
                          margin=dict(t=10,l=10,b=10,r=10))
    return fig

def plotly_subplots_init(rows, cols, specs=None, shared_xaxes=False, shared_yaxes=False, 
                         vertical_spacing=0.05, horizontal_spacing=0.05, font_dict=None,
                         width=1150, height=1000, margin=dict(t=50, b=0, l=0, r=0), title=''):
    fig = go.FigureWidget()
    if font_dict == None:
        font_dict=dict(family='Arial',size=16)
    font_dict['color']=THEME_DICT[THEME]['fontcolor']#'white')
    # specs = [[{} for _ in range(3)] for _ in range(3)]
    # specs[1][0] = {"secondary_y": True}
    # specs[2][0] = {"secondary_y": True}
    sp.make_subplots(rows=rows, cols=cols, figure=fig, specs=specs, 
                     shared_xaxes=shared_xaxes, shared_yaxes=shared_yaxes,
                     vertical_spacing=vertical_spacing, horizontal_spacing=horizontal_spacing)  
    # font_dict=dict(family='Arial',size=16,color='white')
    fig.update_layout(font=font_dict,  # font formatting
                      plot_bgcolor=THEME_DICT[THEME]['bgcolor'],#'black',  # background color
                      template=THEME_DICT[THEME]['plotly'],#'plotly_dark',
                      barmode='overlay', #showlegend=False,
                      width=width, height=height,
                      title=title, #template='plotly_white',  #"plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"
                      margin=dict(t=50, b=0, l=0, r=0)) 

    for i in range(rows):
        for j in range(cols):
            # x and y-axis formatting
            fig.update_yaxes(showline=True,  # add line at x=0
                             showgrid=False,
                             zeroline=False,
                             linecolor=THEME_DICT[THEME]['fontcolor'],#'white',  # line color
                                   linewidth=2.4, # line size
                                   # tickformat="~s" if j>0  or (i==0 and j==0) else None, #ticklabel number format
                                   tickcolor=THEME_DICT[THEME]['fontcolor'],#'white',
                                   # showticklabels=False if j==1 else True,
                                   # ticksuffix = "    " if j==2 else "",
                                   ticks='outside',  # ticks outside axis
                                   # mirror='ticks' if j>0  or (i==0 and j==0) else True,
                                   row=i+1, col=j+1
                            )
            fig.update_xaxes(showline=True,
                             showgrid=False,
                             zeroline=False,
                             linecolor=THEME_DICT[THEME]['fontcolor'],#'white',
                             linewidth=2.4,
                           # tickformat="~s", #ticklabel number format
                             tickcolor=THEME_DICT[THEME]['fontcolor'],#'white',
                             ticks='outside',
                           # mirror='ticks' if j>0  or (i==0 and j==0) else True,
                             row=i+1, col=j+1
                            )
    return fig

def plot_forcevol_histogram(output_df, plot_type='histogram', bins=128, prange=100, psat=100):
#     output_data = {}
#     z_data_temp = data[channel_list[0]][f'Image {img_dir} with Forward Ramps']['data']['Z']
    
#     # z_data_full = np.concatenate([z_data, z_data]) #for both approach and retract for all channels
#     x_len = len(data[channel_list[0]][f'Image {img_dir} with Forward Ramps']['data']['X'])
#     y_len = len(data[channel_list[0]][f'Image {img_dir} with Forward Ramps']['data']['Y'])
#     z_len = len(z_data_temp)
#     # output_data['Z'] = np.reshape([[z_data_full]*(x_len*y_len)], (x_len,y_len,len(z_data_full))).flatten()
#     specdir_list = []
#     z_list = []
#     z_array_dict = {'Forward': z_data_temp, 'Backward': np.flip(z_data_temp)}
#     for spec_dir in ['Forward', 'Backward']:
#         specdir_list.append([SPECT_DICT[spec_dir]]*x_len*y_len*z_len)  
#         z_list.append(np.concatenate([z_array_dict[spec_dir]]*x_len*y_len))
#     # print(z_data, z_list[0][:100])
#     output_data['segment'] = np.concatenate(specdir_list)  
#     output_data['Z'] = np.concatenate(z_list)  
    
#     for chan in channel_list:
#         output_data[chan] = []
#         for spec_dir in ['Forward', 'Backward']:
#             output_data[chan].append(data[chan][f'Image {img_dir} with {spec_dir} Ramps']['data']['ZZ'].flatten(order='F'))
#             # print(len(output_data[chan]), output_data[chan][-1].shape,  len(output_data['segment']), len(output_data['segment'][-1]))
#         output_data[chan] = np.concatenate(output_data[chan])
    
#     output_df = pd.DataFrame(output_data)
    plt.style.use(THEME_DICT[THEME]['matplotlib'])#"dark_background")
    
    output_df_a = output_df[output_df['segment']=='approach']
    output_df_r = output_df[output_df['segment']=='retract']
    #adjust alpha of colormaps
    if THEME == 'dark':
        cmap = plt.cm.Spectral
        BG = np.asarray([0.,0.,0.])
    elif THEME == 'light':
        cmap = plt.cm.coolwarm
        BG = np.asarray([1.,1.,1.])
    my_cmap = cmap(np.arange(cmap.N))
    alphas = np.linspace(0.5,1, cmap.N)
    # alphas = np.logspace(np.log10(0.5),0, cmap.N)
    # alphas = np.ones(cmap.N)
    alphas[0] = 0.2
    alphas[1] = 0.3
    alphas[2] = 0.4
    for i in range(cmap.N):
        my_cmap[i,:-1] = my_cmap[i,:-1]*alphas[i]+BG*(1.-alphas[i])
    my_cmap = ListedColormap(my_cmap)
    plt.close('all')
    if plot_type == 'line':
        # fig, ax = plt.subplots(6,2, figsize = (10, 30))
        fig, ax = plt.subplots(3,2, figsize = (10, 15))
        k = 0
        for i, col_i in enumerate(output_df.columns.drop('segment')):
            print(col_i)
            if col_i != 'Z':
                g = sns.lineplot(data=output_df_a, x='Z', y=col_i, ax=ax[k][0], estimator='median', errorbar=("pi",prange))
                g = sns.lineplot(data=output_df_r, x='Z', y=col_i, ax=ax[k][1], estimator='median', errorbar=("pi",prange))
                ax[k][1].set_ylabel('')
                k += 1

        # ax1[0][0].set_title('approach')
        # ax1[0][1].set_title('retract')
        ax[0][0].set_title('approach')
        ax[0][1].set_title('retract')
        fig_html = fig2html(fig, plot_type='matplotlib', width=900, height=3000, pad=0.1)
    # fig_html2 = fig2html(fig2, plot_type='matplotlib', width=900, height=1500, pad=0.1)
    elif plot_type == 'histogram':
        fig, ax = plt.subplots(6,2, figsize = (10, 30))
        # fig2, ax2 = plt.subplots(3,2, figsize = (10, 15))
        k = 0
        for i, col_i in enumerate(output_df.columns.drop('segment')):          
            for j, col_j in enumerate(output_df.columns.drop('segment')):
                if j > i:
                    print(col_i, col_j)
                    g = sns.histplot(data=output_df_a, x=col_i, y=col_j, bins=bins, ax=ax[k][0], 
                                     pthresh=1-(prange/100), pmax=psat/100,
                                     stat='frequency', edgecolor='none', linewidth=0, cmap=my_cmap)
                    g = sns.histplot(data=output_df_r, x=col_i, y=col_j, bins=bins, ax=ax[k][1], 
                                     pthresh=1-(prange/100), pmax=psat/100,
                                     stat='frequency', edgecolor='none', linewidth=0, cmap=my_cmap)
                    # g = sns.lineplot(data=output_df_a, x=col_i, y=col_j, ax=ax[k][0])
                    # g = sns.lineplot(data=output_df_r, x=col_i, y=col_j, ax=ax[k][1])
                    ax[k][1].set_ylabel('')
                    k += 1

        ax[0][0].set_title('approach')
        ax[0][1].set_title('retract')
        # ax2[0][0].set_title('approach')
        # ax2[0][1].set_title('retract')
        # fig_html1 = fig2html(fig1, plot_type='matplotlib', width=900, height=3000, pad=0.1)
        fig_html = fig2html(fig, plot_type='matplotlib', width=900, height=1500, pad=0.1)
    
    plt.close()
    # plt.show()
    # print('exit')
    return fig_html#, fig_html2


#convert matplotlib/plot plot to html for Jupyter display
def fig2html(fig, plot_type, dpi=300, width=200, height=200, pad=0):
    # Save the plot as binary data
    if plot_type == 'matplotlib':
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=pad, dpi=dpi)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')    
    elif plot_type == 'plotly':
        # buf = fig.to_image(width=size, height=size)
        image_base64 = base64.b64encode(fig.to_image()).decode('ascii')
        
    # Convert the binary data to base64
    # image_base64 = base64.b64encode(buf.read()).decode('utf-8')    
    # Create an HTML image tag
    # '<img src="data:image/png;base64,{}"/>'.format(fig)
    image_tag = f'<img src="data:image/png;base64,{image_base64}" width="{width}" height="{height}"/>'
    return image_tag


#add dashed lines in plot for reference
def plotly_dashedlines(plot_type,fig, x=None, y=None, yaxis=None, visible=True, 
                       line_width=3, secondary_y=False, name=None):
    if plot_type == 'line':
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', yaxis=yaxis, line_dash="dash", 
                                 line_color=THEME_DICT[THEME]['fontcolor'], line_width=line_width,
                                 visible=visible, name=name))
    elif plot_type == 'hline':
        fig.add_hline(y=y, yref=yaxis, secondary_y=secondary_y, line_dash="dash", 
                      line_color=THEME_DICT[THEME]['fontcolor'], line_width=line_width, visible=visible)
    elif plot_type == 'vline':
        fig.add_vline(x=x, line_dash="dash", line_color=THEME_DICT[THEME]['fontcolor'], 
                      line_width=line_width, visible=visible)
    

def merge_plotly_figures(figures, layout):
    """
    Merges Plotly figures into a single image based on the provided layout.

    Parameters:
    figures (list): List of Plotly figures.
    layout (list): List indicating the number of figures to be put horizontally in each row.

    Returns:
    Image: Merged PIL Image object.
    """
    # Convert Plotly figures to images in memory
    images = [PIL.Image.open(io.BytesIO(fig.to_image(format='png'))) for fig in figures]

    # Determine the layout and maximum combined width for each row
    row_heights = []
    row_images = []
    row_widths = []

    index = 0
    for num_cols in layout:
        current_row_images = images[index:index + num_cols]
        row_images.append(current_row_images)
        row_width = sum(img.width for img in current_row_images)
        row_widths.append(row_width)
        index += num_cols

    # Calculate the maximum combined width for any row
    max_combined_width = max(row_widths)

    # Resize images to fit within the maximum combined width while maintaining aspect ratio
    resized_images = []
    for row in row_images:
        resized_row = []
        combined_row_width = sum(img.width for img in row)
        width_ratio = max_combined_width / combined_row_width
        for img in row:
            new_width = int(img.width * width_ratio)
            new_height = int(img.height * width_ratio)
            resized_img = img.resize((new_width, new_height), PIL.Image.LANCZOS)
            resized_row.append(resized_img)
        resized_images.append(resized_row)

    # Calculate the total height of the combined image
    total_height = sum(max(img.height for img in row) for row in resized_images)

    # Create a new blank image with the combined size
    # css_colors = {'white': (255, 255, 255), 'black': (0, 0, 0)}
    combined_image = PIL.Image.new('RGB', (max_combined_width, total_height), (255,255,255))
                                   #css_colors[THEME_DICT[THEME]['bgcolor']])

    # Paste the images into the combined image
    y_offset = 0
    for row in resized_images:
        x_offset = 0
        row_height = max(img.height for img in row)
        for img in row:
            combined_image.paste(img, (x_offset, y_offset))
            x_offset += img.width
        y_offset += row_height

    return combined_image
