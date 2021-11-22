from Opportunities import Opportunities

import datetime


class Lead:
    def __init__(self, source: Opportunities, contract: float, history: {datetime.date: str} = None):
        self.opportunity_source = source
        self.contract = contract
        self.history = {} if history is None else history
