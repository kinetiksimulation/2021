import random

import pandas as pd

from Stage import Stage
from OpportunitySource import OpportunitySource
from WinLoss import Win, Loss

import datetime


class Model:
    def __init__(self, stages: [Stage], sources: [OpportunitySource],
                 start_week: datetime.date = datetime.datetime.now(),
                 wins: Win = Win(), losses: Loss = Loss(),
                 history: {datetime.date: float} = None):
        self.stages = stages
        self.sources = sources
        self.current_week = start_week
        self.history = {} if history is None else history
        self.wins = wins
        self.losses = losses

    def assign_new_leads(self, prob_matrix: [[float]]):
        for stage in self.stages:
            if self.current_week not in stage.history.keys():
                stage.history[self.current_week] = []
            for lead in stage.leads:
                if lead not in stage.history[self.current_week]:
                    stage.history[self.current_week].append(lead)
                    lead.history[self.current_week] = stage.name

        # To treat each probability as independent:
        # for source in range(len(prob_matrix)):
        #     for stage in range(len(prob_matrix[0])):
        #         for lead in self.sources[source].leads:
        #             rand = random.random()
        #             if rand < prob_matrix[source][stage]:
        #                 self.stages[stage].history[self.current_week] += 1
        #                 self.stages[stage].leads.append(lead)
        #                 self.sources[source].leads.remove(lead)

        # To treat probabilities as weights:
        for source, row in enumerate(prob_matrix):
            for lead in self.sources[source].leads:
                stage = random.choices(population=self.stages,
                                       weights=row,
                                       k=1)[0]
                stage.history[self.current_week].append(lead)
                lead.history[self.current_week] = stage.name
                stage.leads.append(lead)
            self.sources[source].leads.clear()

    # TODO - implement probabilities for specific lead's movement
    def update_stages(self, prob_matrix: [[float]], step: int = 1):
        week = self.current_week + datetime.timedelta(weeks=step)
        for stage in self.stages:
            stage.history[week] = []
            # for lead in stage.leads:
            #     stage.history[week].append(lead)
            #     lead.history[week] = stage.name

        # To treat each probability as independent:
        # for source in range(len(prob_matrix)):
        #     for destination in range(len(prob_matrix[0])):
        #         for lead in self.stages[source].leads:
        #             rand = random.random()
        #             if rand < prob_matrix[source][destination]:
        #                 self.stages[destination].leads.append(lead)
        #                 self.stages[source].leads.remove(lead)

        # To treat probabilities as weights:
        remove_leads = []
        for source, row in enumerate(prob_matrix):
            for lead in self.stages[source].leads:
                stage = random.choices(population=self.stages,
                                       weights=row,
                                       k=1)[0]
                if stage.name != self.stages[source].name:
                    stage.leads.append(lead)
                    remove_leads.append(lead)
            for lead in remove_leads:
                self.stages[source].leads.remove(lead)
            remove_leads.clear()

        for stage in self.stages:
            for lead in stage.leads:
                stage.history[week].append(lead)
                lead.history[week] = stage.name

    def calculate_new_revenue(self) -> float:
        revenue = 0.0
        for lead in self.wins.leads:
            revenue += lead.contract
        self.wins.leads.clear()
        self.losses.leads.clear()
        return revenue

    def calculate_renewed_revenue(self, prob: float) -> float:
        k = self.current_week - datetime.timedelta(weeks=52)
        if k in self.history.keys():
            if random.random() < prob:
                return self.history[k]
        return 0.0

    def calculate_total_revenue(self, prob: float) -> float:
        return self.calculate_new_revenue() + self.calculate_renewed_revenue(prob)

    def step(self, contract: float, new_leads_mat: [[float]], update_stages_mat: [[float]], renew_prob: float,
             num_weeks: int = 1) -> float:
        for s in self.sources:
            s.calculate_new_leads(self.current_week, contract)
        self.assign_new_leads(new_leads_mat)
        self.update_stages(update_stages_mat, step=num_weeks)
        self.wins.determine_wins(self.stages, self.current_week + datetime.timedelta(weeks=num_weeks))
        self.losses.determine_losses(self.stages, self.current_week + datetime.timedelta(weeks=num_weeks))
        revenue = self.calculate_total_revenue(renew_prob)
        self.history[self.current_week] = revenue
        self.current_week += datetime.timedelta(weeks=num_weeks)
        return revenue

    def get_report(self) -> pd.DataFrame:
        d = {}
        for stage in self.stages[:-2]:
            d_2 = {}
            for week in self.stages[0].history.keys():
                d_2[week] = len(stage.history[week])
            d[stage.name] = d_2
        df = pd.DataFrame(d)
        df.index.name = "Date"

        d = {}
        for stage in self.stages[-2:]:
            d_2 = {}
            for i, week in enumerate(self.stages[0].history.keys()):
                d_2[week] = len(stage.history[week])
                if i > 0:
                    d_2[week] += d_2[week - datetime.timedelta(weeks=1)]
            d[stage.name] = d_2
        df['Win'] = d["Win"].values()
        df['Loss'] = d["Loss"].values()

        #set up dictionary for the total contract amount per week/stage (could revise these to do it all at once in the future)
        sources = {}
        for stage in self.stages[:-2]:
            d = {}
            for week in stage.history.keys():
                d[week] = []
                for lead in stage.history[week]:
                    d[week].append(lead.opportunity_source.value)
            sources[stage.name] = d
        
        #set up the dictionary for the total contract amount per week/stage
        amounts = {}
        for stage in self.stages:
            d = {}
            for week in stage.history.keys():
                d[week] = 0
                for lead in stage.history[week]:
                    d[week] += lead.contract
            amounts[stage.name] = d
        return (df,sources,amounts)
