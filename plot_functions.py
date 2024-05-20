import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
import numpy as np

# create plot with multiple secondary y axes, grouped by data columns specified in line_group, symbol and line_dash   
# secondary y axis created based on values populated in multiy_col column of data. yvars list can be used to limit the number of
#secondary y axis created. x, y, line_group, symbol, hover_name and line_dash are passed to px.line (check plotly documentation for that)
def plotly_multiyplot(data, multiy_col, yvars, x, y, fig=None, yax_dict=None,
                      line_group=None, symbol=None, line_dash=None, hover_name=None):
    color_list = ['blue', 'red', 'green', 'purple']
    font_dict=dict(family='Arial', size=22, color='black')
    
    if fig == None:
        fig = go.FigureWidget()
        fig.update_layout(font=font_dict,  # font formatting
                          plot_bgcolor='white',  # background color
                          height=500, width=1100, title_text="",
                          margin=dict(t=50, b=0, l=0, r=0),
                          showlegend=False)
    
    if yax_dict == None:
        yax_dict = {}
        
    yvars_old = list(yax_dict.keys())
  
    if all(var in yvars for var in yvars_old) == False:
        yax_dict.clear()
        yvars_new = yvars
        i = 0
    elif list(yvars) == yvars_old:
        fig.data = []
        yvars_new = yvars
        i = 0
    else:
        i = len(yvars_old)
        yvars_new = [yvar_i for yvar_i in yvars if yvar_i not in yvars_old]

    for yvars_i in yvars_new:
        data_i = data[data[multiy_col]==yvars_i]
        trace_i = px.line(data_i, x=x, y=y, line_group=line_group, symbol=symbol, 
                          hover_name=hover_name, line_dash=line_dash, symbol_sequence = ['circle']) 
        trace_i.update_traces(yaxis=f'y{i+1}', line_color=color_list[i],
                              marker=dict(size=1))
        for trace_data_i in trace_i.data:
            fig.add_trace(trace_data_i)
        yax_dict[yvars_i] = f'y{i+1}'
        i += 1
    
    fig.update_xaxes(domain=[0.15, 0.85], title_text='Z',
                     showline=True,
                     linecolor='black',
                     linewidth=2.4,
                     ticks='outside',
                     tickfont=font_dict,
                     title_font=font_dict,
                     tickwidth=2.4,
                     tickcolor='black')
    
    return fig, yax_dict

# delete and recreate secondary y axes based on a given yvars and initialise the plot. yax_dict is a dictionary of previously made
# secondary y axis in the plot, which is compare to create new y axes or recreate everything.
def plotly_multiyplot_initax(fig, yvars, yax_dict):
    color_list = ['blue', 'red', 'green', 'purple']
    font_dict=dict(family='Arial',size=22,color='black')
    
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
            
#plotly version of seaborn's pairplot to plot relational xy data in a grid for all cols in data. Diagonal of the gird shows histogram, use
#nbins to adjust histogram bins
def plotly_pairplot(data, fig=None, cols=None, hue=None, diag_kind='hist', nbins = 50):
    font_dict=dict(family='Arial',
               size=22,
               color='black'
           )
    if cols is None:
        cols = data.columns
    num_cols = len(cols)
    
    if fig == None:
        fig = go.FigureWidget()
        plotly_pairplot_initax(fig, num_cols)
        
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
def plotly_pairplot_initax(fig, num):
    font_dict=dict(family='Arial',size=22,color='black')
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
                  plot_bgcolor='white',  # background color
                  barmode='overlay', showlegend=True, 
                  width=1100, height=1100,
                  title='', #template='plotly_white',  #"plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"
                  margin=dict(t=50, b=0, l=0, r=0))  # Adjust margins to remove white space

    for i in range(num):
        for j in range(num):
            # x and y-axis formatting
            fig.update_yaxes(showline=True,  # add line at x=0
                             linecolor='black',  # line color
                             linewidth=2.4, # line size
                             ticks='outside',  # ticks outside axis
                             tickfont=font_dict, # tick label font
                             title_font=font_dict,
                             tickwidth=2.4,  # tick width
                             tickcolor='black',  # tick color
                             row=i+1, col=j+1, secondary_y=False)
            fig.update_xaxes(showline=True,
                             #showticklabels=True,
                             linecolor='black',
                             linewidth=2.4,
                             ticks='outside',
                             tickfont=font_dict,
                             title_font=font_dict,
                             tickwidth=2.4,
                             tickcolor='black',
                             row=i+1, col=j+1)
            if i==j:
                fig.update_layout({f'yaxis{(num+2)*i+2}': dict(showgrid=False,     # Hide grid lines
                                                        zeroline=False,     # Hide zero line
                                                        showline=False,     # Hide axis line
                                                        ticks='',           # Hide ticks
                                                        showticklabels=False, # Hide tick labels
                                                        title=''
                                                    )})