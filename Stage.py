from Lead import Lead
import datetime


class Stage:
    def __init__(self, name: str, leads: [Lead] = None, history: {datetime.date: [Lead]} = None):
        if history is None:
            history = {}
        self.name = name
        self.leads = [] if leads is None else leads
        self.history = {} if history is None else history
