import datetime
import random
from pandas.tseries.offsets import Week

from tqdm import tqdm

from Model import Model
from Stage import Stage
from OpportunitySource import OpportunitySource
from Distribution import Normal
from Opportunities import Opportunities

from ipywidgets import interact
from plotly.subplots import make_subplots
import plotly.express as px

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly

from collections import Counter


def normalize_rand_prob_matrix(prob_mat: [[float]]) -> [[float]]:
    p = prob_mat
    for i, row in enumerate(p):
        p[i] = (row - row.min()) / (row - row.min()).sum()
    return p
    

#function to create subplots
def make_bar_charts(dataset, sources, amounts, categrical_col, title , frame_rate = 1):

    #main dataset
    main_dataset = dataset.iloc[:, 0:len(dataset.columns)-2]
    #win/loss dataset
    wins_dataset = dataset.iloc[:,len(dataset.columns)-2: len(dataset.columns)]

    #dictionaries for formatting data
    main_cleaned_dict = {'Week': [], 'Stage': [], 'Leads': []}
    wins_cleaned_dict = {'Week': [], 'Type': [], 'Amount': []}
    sources_cleaned_dict = {'Week': [], 'Stage': [], 'Sources': []}
    amounts_cleaned_dict = {'Week': [], 'Stage': [], 'Amount': []}

    #formatting the data
    for i in range(len(main_dataset.index)):
        for x in range(len(main_dataset.columns)):
            main_cleaned_dict['Week'].append(str(list(main_dataset.index)[i]))
            main_cleaned_dict['Stage'].append(list(main_dataset.columns)[x])
            main_cleaned_dict['Leads'].append(main_dataset.iloc[i][list(main_dataset.columns)[x]])
        for k in range(len(wins_dataset.columns)):
            wins_cleaned_dict['Week'].append(str(list(wins_dataset.index)[i]))
            wins_cleaned_dict['Type'].append(list(wins_dataset.columns)[k])
            wins_cleaned_dict['Amount'].append(wins_dataset.iloc[i][list(wins_dataset.columns)[k]])
        for j in range(len(sources.columns)):
            sources_cleaned_dict['Week'].append(str(list(sources.index)[i]))
            sources_cleaned_dict['Stage'].append(list(sources.columns)[j])
            sources_cleaned_dict['Sources'].append(sources.iloc[i][list(sources.columns)[j]])
        for l in range(len(amounts.columns)):
            amounts_cleaned_dict['Week'].append(str(list(amounts.index)[i]))
            amounts_cleaned_dict['Stage'].append(list(amounts.columns)[l])
            amounts_cleaned_dict['Amount'].append(amounts.iloc[i][list(amounts.columns)[l]])

    #creating dfs
    main_df = pd.DataFrame(main_cleaned_dict)
    wins_df = pd.DataFrame(wins_cleaned_dict)
    sources_df = pd.DataFrame(sources_cleaned_dict)
    amounts_df = pd.DataFrame(amounts_cleaned_dict)
    sources_amounts_df = pd.merge(sources_df,amounts_df,how='inner',on=('Week','Stage'))
    
    #start and end weeks for dataset, currently not being used
    start_week = str(main_df.Week.min())
    end_week = str(main_df.Week.max())

    #parent figure
    fig = make_subplots(3, 3, subplot_titles=('','Leads per Week by Stage','','','','','','Results for Each Week',''))

    #color formatting
    main_bar_colors = ['#78a9ff','#4589ff','#0f62fe','#0043ce','#002d9c','#001d6c']
    wins_bar_colors = ['#0e6027','#a2191f']

    #create plots for each week.
    for i in range(len(main_df.Week.unique())):
        date = main_df.Week.unique()[i]

        #default value for which plots are visible
        visible = False

        #setting first plots only to visible
        if i == 0:
            visible = True

        #total contract amounts for each stage
        amounts_list = [x for x in sources_amounts_df[sources_amounts_df.Week == date]['Amount']]

        #top middle bar
        fig.add_trace(go.Bar(x=sources_amounts_df[sources_amounts_df.Week == date]['Stage'],y=[len(x) for x in sources_amounts_df[sources_amounts_df.Week == date]['Sources']],visible=visible, marker_color=main_bar_colors,hovertemplate='<b>Leads</b>: %{y}' + '<br>%{customdata}<extra></extra>',customdata=['<b>Total Amount for Stage</b>: {}'.format(x) for x in amounts_list]),row=1,col=2)
        
        #bottom middle bar
        fig.add_trace(go.Bar(y=wins_df[wins_df.Week == date]['Type'],x=wins_df[wins_df.Week == date]['Amount'],visible=visible,orientation='h',marker_color=wins_bar_colors,hovertemplate=['Wins: %{x}<extra></extra>','Losses: %{x}<extra></extra>']),row=3,col=2)
 
    
    #creating steps for sliders, each step makes the plots for that week visible
    steps = []
    for i in range(len(main_df.Week.unique())):
        date = main_df.Week.unique()[i]
        step = dict(
            method = 'restyle',  
            args = ['visible', [False] * len(fig.data)],
            label = str(date)
        )
        step['args'][1][i*2] = True
        step['args'][1][i*2+1] = True
        steps.append(step)

    #creating sliders
    sliders = [dict(
        steps = steps,
        font=dict(
            size=18
        ),
        bgcolor='black',
        pad={'t':50,'b':50},
        y=0.6,
        currentvalue={'visible':True, 'prefix': 'Week: '}
    )]

    #setting layout props
    fig.layout.sliders = sliders
    fig.layout.yaxis2.range = [0,main_df.Leads.max()+1]
    fig.layout.xaxis8.range = [0,wins_df.Amount.max()+1]
    fig.layout.xaxis2.title = 'Stage'
    fig.layout.xaxis8.title = 'Amount'
    fig.layout.yaxis2.title = 'Leads'
    fig.layout.yaxis8.title = 'Results'
    fig.layout.showlegend = False

    #go.FigureWidget(fig)

    fig.show()

def plot_history(m: Model):
    # Get model report
    dataset = m.get_report()

    sources = pd.DataFrame(dataset[1])
    amounts = pd.DataFrame(dataset[2])
    dataset = dataset[0]

    # print(df)
    # Ignore Win/Loss column - Uncomment to show
    #
    make_bar_charts(dataset, sources, amounts, "Stage", title = "Leads", frame_rate = 5)    


def main():
    # Set parameters
    num_weeks = 52
    # random.seed(10)

    # Create model stages
    stages = [
        Stage("Engage"),
        Stage("Qualify"),
        Stage("Design"),
        Stage("Propose"),
        Stage("Negotiate"),
        Stage("Close"),
        Stage("Win"),
        Stage("Loss")
    ]

    # n = Normal(std_dev=0, mean=0)
    date = datetime.date(year=2021, month=11, day=15)

    # Create model sources
    sources = [
        OpportunitySource(dist=Normal(std_dev=1.0, mean=0.5),
                          opportunity_type=Opportunities.TIGER_TEAM),
        OpportunitySource(dist=Normal(std_dev=0.1, mean=0.001),
                          opportunity_type=Opportunities.OTHER_SALES),
        OpportunitySource(dist=Normal(std_dev=0.1, mean=0.001),
                          opportunity_type=Opportunities.MARKETING),
        OpportunitySource(dist=Normal(std_dev=2.0, mean=1.25),
                          opportunity_type=Opportunities.CURRENT_AI_SELLERS),
        OpportunitySource(dist=Normal(std_dev=0.1, mean=0.001),
                          opportunity_type=Opportunities.BUSINESS_PARTNERS),
        OpportunitySource(dist=Normal(std_dev=0.1, mean=0.001),
                          opportunity_type=Opportunities.DIGITAL_DEMAND),
        OpportunitySource(dist=Normal(std_dev=0.1, mean=0.001),
                          opportunity_type=Opportunities.CONSULTING)
    ]

    # Create model history
    new_date = date - datetime.timedelta(weeks=52)
    history = {}
    for _ in range(52):
        history[new_date] = 200000.0
        new_date += datetime.timedelta(weeks=1)

    # Initialize model
    m = Model(
        stages=stages,
        sources=sources,
        start_week=date,
        history=history
    )

    # new_leads = np.random.rand(len(sources), len(stages))
    # new_leads = normalize_rand_prob_matrix(new_leads)
    # update_stages = np.random.rand(len(stages), len(stages))
    # update_stages = normalize_rand_prob_matrix(update_stages)

    # Probability matrix for new leads
    new_leads = [
        [0.45, 0.4, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.45, 0.4, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.45, 0.4, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.45, 0.4, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.45, 0.4, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.45, 0.4, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.45, 0.4, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]
    new_leads = np.array(new_leads)

    # Probability matrix for movement between stages
    update_stages = [
        [0.9, 0.03, 0.03, 0.01, 0.01, 0.01, 0.0, 0.01],
        [0.01, 0.9, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01],
        [0.01, 0.01, 0.9, 0.04, 0.02, 0.02, 0.0, 0.0],
        [0.01, 0.01, 0.01, 0.89, 0.03, 0.03, 0.01, 0.01],
        [0.01, 0.01, 0.01, 0.01, 0.91, 0.3, 0.01, 0.01],
        [0.0, 0.0, 0.0, 0.0, 0.01, 0.9, 0.03, 0.06]
    ]
    update_stages = np.array(update_stages)

    # Run model for N weeks
    for _ in range(num_weeks):
        m.step(
            contract=120000.0,
            new_leads_mat=new_leads,
            update_stages_mat=update_stages,
            renew_prob=85.0
        )

    # Calculate revenue for this trial and store in revenue list
    revenue = 0.0
    for i in range(num_weeks):
        revenue += m.history[date]
        date += datetime.timedelta(weeks=1)
    print("Total Revenue: " + str(revenue))

    plot_history(m)


if __name__ == "__main__":
    main()
