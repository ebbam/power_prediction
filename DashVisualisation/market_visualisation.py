
from market_functions import *
from processing_data import sample_parameters
from plotly_plots import *
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px


# Variables of interest

# get colour list
variables_of_interest = ['budget', 'risk_av', 'stubbornness', 'expertise']
color_list = px.colors.qualitative.Plotly

nice_names = ['Initial Budget', 'Risk aversion', 'Stubbornness', 'Expertise']

parameters = {'n_bettors': 100, # The number of betting agents
              #'el_outcome': 1, # Q: Ultimate election outcome - assuming we know this to begin with and it does not change over time...for now this is implemented as a random walk of the probability...but should this be 0 or 1 instead? '''
              't_election': 100, # Time until election takes place (ie. time horizon of betting)
              'initial_price': 0.5, # Initial market price (is this equivalent to probability of winning)
              'outcome_uncertainty': 0.1} # This is a measure of how uncertain the true outcome is - ie. the volatility of the random walk election probability

bettor_attribute_colours = {variables_of_interest[i]: color_list[i] for i in range(len(variables_of_interest))}

color_list = color_list[len(variables_of_interest):]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container([
    html.H1("Market Simulation Dashboard"),
    # html.Br(),

    dbc.Row([
        dbc.Col(html.H3("Bettor attributes"),width=3),
        dbc.Col(html.H3("Mean"),width=5),
        dbc.Col(width=1),
        dbc.Col(html.H3("Variance"),width=3),
    ]),
    html.Br(),
    html.Br(),
    # Input fields for budget, risk_av, stubbornness, expertise
    dbc.Row([
        dbc.Col(html.Div(nice_names[0], style={'color':bettor_attribute_colours[variables_of_interest[0]]}),width=3),
        dbc.Col(dcc.Slider(500, 1500, 100,
                marks = {i : str(i) for i in range(500,1501,200)},
               value=1000, id="budget_mean"),width=5),
        dbc.Col(width=1),
        dbc.Col(dcc.Input(id="budget_var", type="number", value=100),width=3),
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col(html.Div(nice_names[1],style={'color':bettor_attribute_colours[variables_of_interest[1]]}),width=3),
        dbc.Col(dcc.Slider(0, 1, 0.1,
               value=0.5, id="risk_av_mean"),width=5),
               dbc.Col(width=1),
        dbc.Col(dcc.Input(id="risk_av_var", type="number", value=0.01, min = 0, max = 0.4),width=3),
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col(html.Div(nice_names[2],style={'color':bettor_attribute_colours[variables_of_interest[2]]}),width=3),
        dbc.Col(dcc.Slider(0, 1, 0.1,
               value=0.5, id="stubbornness_mean"),width=5),
        dbc.Col(width=1),
        dbc.Col(dcc.Input(id="stubbornness_var", type="number", value=0.01, min = 0, max = 0.4),width=3),
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col(html.Div(nice_names[3],style={'color':bettor_attribute_colours[variables_of_interest[3]]}),width=3),
        dbc.Col(dcc.Slider(0, 1, 0.9,
               value=0.5, id="expertise_mean"),width=5),
        dbc.Col(width=1),
        dbc.Col(dcc.Input(id="expertise_var", type="number", value=0.01, min = 0, max = 0.4),width=3),
    ]),

    # add some vertical space
    html.Br(),
    html.Br(),
    # Dropdowns for variable correlation selection and correlation input
    dbc.Row([dbc.Col(html.Div('Correlate '),width=1),
        dbc.Col(dcc.Dropdown(
            id="var_1",
            options=[{"label": nice_names[i], "value": variables_of_interest[i]} for i in range(len(variables_of_interest))],
            value="budget"
        ),width=3),
        dbc.Col(html.Div(' and '),width=1),
        dbc.Col(dcc.Dropdown(
            id="var_2",
            options=[{"label": nice_names[i], "value": variables_of_interest[i]} for i in range(len(variables_of_interest))],
            value="expertise"
        ),width=3),
        dbc.Col(html.Div(' with correlation coefficient  '),width=3),
        dbc.Col(dcc.Input(id="correlation", type="number", value=0),width=1),
    ]),

    
    # Plots
    dbc.Row([
        dbc.Col(dcc.Graph(id="hist_plot_1"), width=4),
        dbc.Col(dcc.Graph(id="hist_plot_2"), width=4),
        dbc.Col(dcc.Graph(id="joint_plot"), width=4),
    ],className="g-0"),

    html.Br(),

    dbc.Row([
        dcc.Graph(id="market_price_plot")]),
    dbc.Row([html.Br()]),
    dbc.Row([
        dcc.Graph(id="supply_demand_plot")]),
    dbc.Row([html.Br()]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="market_error_plot")),
    ])
])

# Callbacks
@app.callback(
    [Output("hist_plot_1", "figure"),
     Output("hist_plot_2", "figure"),
     Output("joint_plot", "figure"),
     Output("market_price_plot", "figure"),
     Output("supply_demand_plot", "figure"),
     Output("market_error_plot", "figure")],
    [
    Input("var_1", "value"),
    Input("var_2", "value"),
    Input("correlation", "value"),
    Input("budget_mean", "value"), 
    Input("budget_var", "value"),
    Input("risk_av_mean", "value"), 
    Input("risk_av_var", "value"),
    Input("stubbornness_mean", "value"),
    Input("stubbornness_var", "value"),
    Input("expertise_mean", "value"), 
    Input("expertise_var", "value")]
)
def update_plots(var_1, var_2, correlation, budget_mean, budget_var,
                 risk_av_mean, risk_av_var, stubbornness_mean, stubbornness_var,
                 expertise_mean, expertise_var):
    
    # Prepare parameters from user inputs
    means = [budget_mean, risk_av_mean, stubbornness_mean, expertise_mean]
    sds = [budget_var, risk_av_var, stubbornness_var, expertise_var]
    
    var_1_ind = variables_of_interest.index(var_1)
    var_2_ind = variables_of_interest.index(var_2)

    sampled_parameters = sample_parameters(var_1_ind, var_2_ind, means, sds, correlation, parameters['n_bettors'])

    parameters.update({'bettors': [bettor(budget = sampled_parameters[i,0], 
                                        risk_av = np.clip(sampled_parameters[i,1],0,1), 
                                        stubbornness = np.clip(sampled_parameters[i,2],0,1), 
                                        expertise = np.clip(sampled_parameters[i,3],0,1)) for i in range(parameters['n_bettors'])]})

    bettor_data = {'budget': [k.budget for k in parameters['bettors']],
            'risk_av': [k.risk_av for k in parameters['bettors']],
            'stubbornness': [k.stubbornness for k in parameters['bettors']],
            'expertise': [k.expertise for k in parameters['bettors']]}
    

    other_var = [x for x in variables_of_interest if x not in [var_1, var_2]]
    # Generate plots using your functions
    fig_hist_1 = plot_hist(bettor_data[other_var[0]], bettor_attribute_colours[other_var[0]])
    fig_hist_2 = plot_hist(bettor_data[other_var[1]], bettor_attribute_colours[other_var[1]])
    fig_joint = joint_plot(bettor_data, variables_of_interest, var_1_ind,
                           var_2_ind, bettor_attribute_colours[var_1], bettor_attribute_colours[var_2])

    # Simulate the market
    market_record = run_market(**parameters)

    fig_market_price = plot_market_price(market_record, scale_fact=10, step=True)
    fig_supply_demand = plot_supply_demand(market_record)
    fig_market_error = plot_market_error(market_record, scale_fact=10, step=True)


    return fig_hist_1, fig_hist_2, fig_joint, fig_market_price, fig_supply_demand, fig_market_error




# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
