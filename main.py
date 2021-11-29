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


def normalize_rand_prob_matrix(prob_mat: [[float]]) -> [[float]]:
    p = prob_mat
    for i, row in enumerate(p):
        p[i] = (row - row.min()) / (row - row.min()).sum()
    return p
    

def make_bar_chart(dataset, categrical_col, title , frame_rate = 1):

    main_dataset = dataset.iloc[:, 0:len(dataset.columns)-2]
    wins_dataset = dataset.iloc[:,len(dataset.columns)-2: len(dataset.columns)]
    main_cleaned_dict = {'Week': [], 'Stage': [], 'Leads': []}
    wins_cleaned_dict = {'Week': [], 'Type': [], 'Amount': []}
    for i in range(len(main_dataset.index)):
        for x in range(len(main_dataset.columns)):
            main_cleaned_dict['Week'].append(str(list(main_dataset.index)[i]))
            main_cleaned_dict['Stage'].append(list(main_dataset.columns)[x])
            main_cleaned_dict['Leads'].append(main_dataset.iloc[i][list(main_dataset.columns)[x]])
        for k in range(len(wins_dataset.columns)):
            wins_cleaned_dict['Week'].append(str(list(wins_dataset.index)[i]))
            wins_cleaned_dict['Type'].append(list(wins_dataset.columns)[k])
            wins_cleaned_dict['Amount'].append(wins_dataset.iloc[i][list(wins_dataset.columns)[k]])


            
        


    main_df = pd.DataFrame(main_cleaned_dict)
    wins_df = pd.DataFrame(wins_cleaned_dict)

    start_week = str(main_df.Week.min())
    end_week = str(main_df.Week.max())

    fig = make_subplots(3, 1, subplot_titles=('Leads per Week by Stage','','Results for Each Week'))

    main_bar_colors = ['#78a9ff','#4589ff','#0f62fe','#0043ce','#002d9c','#001d6c']
    wins_bar_colors = ['#0e6027','#a2191f']
    for i in range(len(main_df.Week.unique())):
        date = main_df.Week.unique()[i]
        visible = False
        if i == 0:
            visible = True
        fig.add_bar(x=main_df[main_df.Week == date]['Stage'],y=main_df[main_df.Week == date]['Leads'],visible=visible,row=1,col=1, marker_color=main_bar_colors,hovertemplate='Leads: %{y}<extra></extra>')
        fig.add_bar(y=wins_df[wins_df.Week == date]['Type'],x=wins_df[wins_df.Week == date]['Amount'],visible=visible,row=3,col=1,orientation='h',marker_color=wins_bar_colors,hovertemplate=['Wins: %{x}<extra></extra>','Losses: %{x}<extra></extra>'])
 
    
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
    
    fig.layout.sliders = sliders
    fig.layout.yaxis.range = [0,main_df.Leads.max()+1]
    fig.layout.xaxis3.range = [0,wins_df.Amount.max()+1]
    fig.layout.xaxis.title = 'Stage'
    fig.layout.xaxis3.title = 'Amount'
    fig.layout.yaxis.title = 'Leads'
    fig.layout.yaxis3.title = 'Results'

    fig.layout.showlegend = False

    go.FigureWidget(fig)

    fig.show()

def plot_history(m: Model):
    # Get model report
    dataset = m.get_report()
    # print(df)
    # Ignore Win/Loss column - Uncomment to show
    #
    
    make_bar_chart(dataset, "Stage", title = "Leads", frame_rate = 5)    


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
