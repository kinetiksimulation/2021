from Distribution import Distribution
from Opportunities import Opportunities
from Lead import Lead

import datetime


class OpportunitySource:
    def __init__(self, dist: Distribution,
                 opportunity_type: Opportunities,
                 leads: [Lead] = None,
                 history: {datetime.date: [Lead]} = None):
        self.dist = dist
        self.history = {} if history is None else history
        self.leads = [] if leads is None else leads
        self.opportunity_type = opportunity_type

    def calculate_new_leads(self, week: datetime.date, contract: float, dist: Distribution = None):
        if dist is None:
            dist = self.dist
        new_leads = round(dist.inverse())
        # new_leads = 10000
        self.history[week] = []
        for _ in range(new_leads):
            lead = Lead(self.opportunity_type, contract)
            self.leads.append(lead)
            self.history[week].append(lead)
