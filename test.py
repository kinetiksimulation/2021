import datetime
import random

from tqdm import tqdm

from Model import Model
from Stage import Stage
from OpportunitySource import OpportunitySource
from Distribution import Normal
from Opportunities import Opportunities

from ipywidgets import interact
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


def plot_history(m: Model):
    # Get model report
    df = m.get_report()
    # print(df)

    # Ignore Win/Loss column - Uncomment to show
    df = df.iloc[:, 0:len(df.columns)-2]

    # Find max leads for any given stage for any given week to lock y-axis
    max_val = 0
    for col in df.columns:
        max_val = max(df[col].max(), max_val)

    # Generate the plot
    fig = go.Figure(layout_yaxis_range=[0, max_val + 1])
    for step in range(len(df.index)):
        fig.add_trace(
            go.Bar(
                visible=False,
                x=df.columns.values,
                y=df.iloc[step].values,
            ))
    fig.data[0].visible = True

    # Set the steps
    steps = []
    for i, date in enumerate(df.index):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": str(date)}],  # layout attribute
            label=str(date),
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    # Create sliders
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Date: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )

    # Show figure
    fig.show()


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
