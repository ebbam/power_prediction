import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def plot_market_price(rec, scale_fact=10, step=True, colors=None):
    # Default colors if none are provided
    if colors is None:
        colors = ["black", "darkorange", "dodgerblue", "red", "green", "purple"]

    # Process market pressure
    mp = np.array(rec['market_pressure'])[1:]
    upward_pressure = rec['price_history'][:-1] + ([mp > 0] * mp)[0] / scale_fact  # positive market pressure
    downward_pressure = rec['price_history'][:-1] + ([mp < 0] * mp)[0] / scale_fact  # negative market pressure
    N = len(rec["price_history"])

    # Create the subplots
    fig = go.Figure()

    # Market price, upward and downward pressure
    if step:
        fig.add_trace(go.Scatter(x=list(range(N)), y=rec['price_history'], mode='lines',
                                 line=dict(color=colors[0], shape='hv', width=3), name="Market price"))
        
        fig.add_trace(go.Scatter(x=list(range(N - 1)), y=upward_pressure, mode='lines',
                                 line=dict(color=colors[1],shape='hv', width = 0), fill='tonexty', name="Upward pressure",
                                 fillcolor=colors[1], opacity=0.2))
        
        fig.add_trace(go.Scatter(x=list(range(N)), y=rec['price_history'], mode='lines',
                            line=dict(color=colors[0], shape='hv',width=3),showlegend = False))
        
        fig.add_trace(go.Scatter(x=list(range(N - 1)), y=downward_pressure, mode='lines',
                                 line=dict(color=colors[2],shape='hv', width = 0), fill='tonexty', name="Downward pressure",
                                 fillcolor=colors[2], opacity=0.2))

        fig.add_trace(go.Scatter(x=list(range(N)), y=rec['gen_el'], mode='lines',
                                 line=dict(color=colors[3], shape='hv', width=1), name="Election outcome"))

        fig.add_trace(go.Scatter(x=list(range(N)), y=rec['beliefs'], mode='lines',
                                 line=dict(color=colors[4], dash='2px', shape='hv', width=1), name="Average market valuation"))
        
        fig.add_trace(go.Scatter(x=list(range(N)), y=rec['weighted_beliefs'], mode='lines',
                            line=dict(color=colors[5], dash='1px', shape='hv', width=1), name="Weighted average market valuation"))
    else:
        fig.add_trace(go.Scatter(x=list(range(N)), y=rec['price_history'], mode='lines',
                                 line=dict(color=colors[0], width=3), name="Market price"))

        fig.add_trace(go.Scatter(x=list(range(N - 1)), y=upward_pressure, mode='lines',
                                 line=dict(color=colors[1], width=0), fill='tonexty', name="Upward pressure",
                                 fillcolor=colors[1], opacity=0.5))
        
        fig.add_trace(go.Scatter(x=list(range(N)), y=rec['price_history'], mode='lines',
                            line=dict(color=colors[0], width=3),showlegend = False))
        
        fig.add_trace(go.Scatter(x=list(range(N - 1)), y=downward_pressure, mode='lines',
                                 line=dict(color=colors[2],width=0), fill='tonexty', name="Downward pressure",
                                 fillcolor=colors[2], opacity=0.5))

        fig.add_trace(go.Scatter(x=list(range(N)), y=rec['gen_el'], mode='lines',
                                 line=dict(color=colors[3], width=1), name="Election outcome"))

        fig.add_trace(go.Scatter(x=list(range(N)), y=rec['beliefs'], mode='lines',
                                 line=dict(color=colors[4], dash='2px', width=1), name="Average market valuation"))
        
        fig.add_trace(go.Scatter(x=list(range(N)), y=rec['weighted_beliefs'], mode='lines',
                    line=dict(color=colors[5], dash='1px', width=1), name="Weighted average market valuation"))


    # Update layout
    fig.update_layout(
        title="Market Price",
        xaxis_title="Time",
        yaxis=dict(title="Result", range=[0, 1.02]),
        legend=dict(traceorder='normal', x=0, y=-0.2, orientation="h"),
        height=600,
        margin=dict(t=40, b=20),
        template = 'plotly_white'
    )

    return fig



def plot_market_error(rec, scale_fact=10, step=True):

    # Process market pressure
    N = len(rec["price_history"])

    # Create the subplots
    fig = go.Figure()

    # Market price, upward and downward pressure
    if step:
        fig.add_trace(go.Scatter(x=np.arange(N), y=np.zeros(N), mode='lines', opacity=0,
                                 line=dict(color="rgba(0,0,0,0.2)",shape='hv', width = 0),showlegend = False))
        
        fig.add_trace(go.Scatter(x=list(range(N)), y=np.cumsum(np.array(rec['price_history']) - np.array(rec['gen_el'])), mode='lines',
                        line=dict(color="rgba(0,0,0,0.2)",shape='hv', width = 0), fill='tonexty',
                        fillcolor="rgba(0,0,0,0.2)", showlegend = False))
    
        fig.add_trace(go.Scatter(x=np.arange(N), y=np.zeros(N), mode='lines', opacity=0,
                                 line=dict(color="rgba(0,0,0,0.2)",shape='hv', width = 0),showlegend = False))
        
        fig.add_trace(go.Scatter(x=list(range(N)), y=np.array(rec['price_history']) - np.array(rec['gen_el']), mode='lines',
                        line=dict(color="rgba(1,0,0,1)",shape='hv', width = 0), fill='tonexty',
                        fillcolor="rgba(1,0,0,1)", opacity=0.4, showlegend = False))
        

        
    else:
        fig.add_trace(go.Scatter(x=np.arange(N), y=np.zeros(N), mode='lines', opacity=0,
                                 line=dict(color="rgba(0,0,0,0.2)", width = 0),showlegend = False))
        
        fig.add_trace(go.Scatter(x=list(range(N)), y=np.cumsum(np.array(rec['price_history']) - np.array(rec['gen_el'])), mode='lines',
                        line=dict(color="rgba(0,0,0,0.2)", width = 0), fill='tonexty',
                        fillcolor="rgba(0,0,0,0.2)", showlegend = False))
    
        fig.add_trace(go.Scatter(x=np.arange(N), y=np.zeros(N), mode='lines', opacity=0,
                                 line=dict(color="rgba(0,0,0,0.2)", width = 0),showlegend = False))
        
        fig.add_trace(go.Scatter(x=list(range(N)), y=np.array(rec['price_history']) - np.array(rec['gen_el']), mode='lines',
                        line=dict(color="rgba(1,0,0,1)", width = 0), fill='tonexty',
                        fillcolor="rgba(1,0,0,1)", opacity=0.4, showlegend = False))

    # Update layout
    fig.update_layout(
        title="Difference between Market Price and Election Outcome",
        xaxis_title="Time",
        yaxis=dict(title="Result"),
        height=300,
        margin=dict(t=40, b=20),
        template = 'plotly_white'
    )

    return fig


def plot_supply_demand(rec, colors=None):
    # Default colors if none are provided
    if colors is None:
        colors = ["red","blue"]

    N = len(rec["price_history"])

    # Create the subplots
    fig = go.Figure()


    # Volume and net supply
    fig.add_trace(go.Scatter(x=list(range(N)), y=rec['vol_history'], mode='lines',
                             line=dict(color=colors[0], shape='hv'), name="No. of Contracts",
                             yaxis="y2"))

    fig.add_trace(go.Scatter(x=list(range(N)), y=rec['net_supply'], mode='lines',
                             line=dict(color=colors[1], dash='dot', shape='hv'), name="Net Supply",
                             yaxis="y2"))

    # Update layout
    fig.update_layout(
        title="Contract Volume & Net Supply",
        xaxis_title="Time",
        yaxis=dict(title="Result", range=[0, 1]),
        legend=dict(x=0, y=-0.2, orientation="h"),
        height=600,
        margin=dict(t=40, b=20),
        template = 'plotly_white'
    )

    return fig


def plot_hist(data_,c):
    hist_fig = go.Figure(data=[go.Histogram(x=data_,
                                   marker = dict(color=c))])
    hist_fig.update_layout(
        autosize=True,
        minreducedwidth=250,
        minreducedheight=250,
        xaxis = dict(color = 'rgba(0,0,0,0.8)'),
        yaxis = dict(color = 'rgba(0,0,0,0.8)'),
        # height = 600,
        # width = 600,
        bargap = 0,
        hovermode = 'closest',
        showlegend = False,
        template = 'plotly_white'
    )

    return hist_fig


def joint_plot(better_data, variables_of_interest, var_1_ind, var_2_ind, c1, c2):
    joint_plot = go.Figure()#(rows=1, cols=3)

    joint_plot.add_trace(go.Scatter(
            x=better_data[variables_of_interest[var_1_ind]], y=better_data[variables_of_interest[var_2_ind]], 
            xaxis = 'x',
            yaxis = 'y',
            mode = 'markers',
            marker = dict(
                color = 'rgba(0,0,0,0.5)',
                size = 4
            )
        ),# row = 1, col = 1
        )

    # Side dist
    joint_plot.add_trace(go.Histogram(
            y=better_data[variables_of_interest[var_2_ind]], 
            xaxis = 'x2',
            marker = dict(
                color = c1,
            ),
        ),# row = 1, col = 1
        )

    # Side dist
    joint_plot.add_trace(go.Histogram(
            x=better_data[variables_of_interest[var_1_ind]],
            yaxis = 'y2',
            marker = dict(
                color = c2
            )
        ),# row = 1, col = 1
        )

    joint_plot.update_layout(
        autosize=True,
        minreducedwidth=250,
        minreducedheight=250,
        xaxis = dict(
            zeroline = False,
            domain = [0,0.75],
            showgrid = False,
            color = 'rgba(0,0,0,0.8)'
            # dtick = np.round((np.max(better_data[variables_of_interest[var_1_ind]]) - np.min(better_data[variables_of_interest[var_1_ind]]))/5,1)
        ),
        # xaxis_range = [0.90*np.min(better_data[variables_of_interest[var_1_ind]]), 1.1*np.max(better_data[variables_of_interest[var_1_ind]])],
        yaxis = dict(
            zeroline = False,
            domain = [0,0.75],
            showgrid = False,
            color = 'rgba(0,0,0,0.8)'
            # dtick = np.round((np.max(better_data[variables_of_interest[var_2_ind]]) - np.min(better_data[variables_of_interest[var_2_ind]]))/5,1)
        ),
        # yaxis_range = [0.90*np.min(better_data[variables_of_interest[var_2_ind]]), 1.1*np.max(better_data[variables_of_interest[var_2_ind]])],
        xaxis2 = dict(
            zeroline = False,
            domain = [0.8,1],
            showgrid = False,
            color = 'rgba(0,0,0,0.3)'
        ),
        yaxis2 = dict(
            zeroline = False,
            domain = [0.8,1],
            showgrid = False,
            color = 'rgba(0,0,0,0.3)'
        ),
        # height = 600,
        # width = 600,
        bargap = 0,
        hovermode = 'closest',
        showlegend = False,
        template = 'plotly_white'
    )
    return joint_plot