from Stage import Stage
from Lead import Lead
import datetime


class Win(Stage):
    def __init__(self, leads: [Lead] = None, history: {datetime.date: float} = None):
        super(Win, self).__init__("Win", leads=leads, history=history)

    def determine_wins(self, stages: [Stage], current_week: datetime.date):
        for stage in stages:
            if str(stage.name).lower() == "win":
                self.history[current_week] = stage.leads.copy()
                for lead in stage.leads:
                    self.leads.append(lead)
                stage.leads.clear()
                return


class Loss(Stage):
    def __init__(self, leads: [Lead] = None, history: {datetime.date: float} = None):
        super(Loss, self).__init__("Loss", leads=leads, history=history)
        self.history = {} if history is None else history

    def determine_losses(self, stages: [Stage], current_week: datetime.date):
        for stage in stages:
            if str(stage.name).lower() == "loss":
                self.history[current_week] = stage.leads.copy()
                for lead in stage.leads:
                    self.leads.append(lead)
                stage.leads.clear()
                return
