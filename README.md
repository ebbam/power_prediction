# Power and Prediction Markets

*Authors: Ebba Mark, Bridget Smart, Anne Bastian, Josefina Waugh*

This code is associated with the preprint available [here](https://arxiv.org/abs/2601.20452).

This repository contains two project folders (Note: files described below are not an exhaustive list of the repository content):

## code
extensions/: subfolder which includes main_pred_market.ipynb: Jupyter Notebook that demonstrates and describes the functionality of the prediction market model.

simulations/: subfolder that includes code and results for the simulation experiments varying agent attributes and introducing a whale agent on the betting market. 

simulations/bettor.py: bettor class and market functions required to run the model

## DashVisualisation
Contains all relevant files (including market_functions.py - copy of bettor.py) for running a Dash app that allows users to test the performance of the agent-based prediction market model when varying the distribution of agent attribute values in the betting population. 

run "market_visualisation.py" to launch the Dash app on a local host. 

<img alt="dash_highres" src="https://github.com/user-attachments/assets/00b511cd-fb9f-4084-9c50-bc75040dbc3c" />
