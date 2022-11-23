from transformers import AutoModel, AutoTokenizer
import string
from copy import deepcopy


class Debate:
    def __init__(self, num_parties=2, objectives=None, model="distilgpt2", tokenizer=None):
        self.num_parties = num_parties

        if objectives:
            self.objectives = objectives
        else:
            self.objectives = [[1, 0], [0, 1]]

        if isinstance(model, str):
            self.model = AutoModel.from_pretrained(model)

        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif tokenizer == None:
            self.tokenizer = AutoTokenizer.from_pretrained(model)

        self.curr_party = 0
        self.curr_round = 0
        self.num_branches = 1
        self.sel_party = None
        self.sel_round = None
        self.sel_branch = None
        self.aliases = string.ascii_uppercase[:num_parties]
        self.prop_grid = [[[]]] # branch x round x party

    def who(self, party: int):
        assert party < self.num_parties and party >= 0 and type(party) == int, f"Current debate only has {self.num_parties} (zero-indexed) parties. You asked for party {party}, which is unavailable."
        clone = self._clone()
        clone.sel_party = party
        return clone

    def when(self, round: int):
        assert round <= self.curr_round and round >= 0 and type(round) == int, f"Current debate has only been running for {self.curr_round} (zero-indexed) rounds. You asked for round {round}, which hasn't happened yet."
        clone = self._clone()
        clone.sel_round = round
        return clone

    def which(self, branch: int):
        assert branch < self.num_branches and branch >= 0 and type(branch) == int, f"Current debate has only been running across {self.num_branches} (zero-indexed) parallel branches. You asked for branch {branch}, which is unavailable."
        clone = self._clone()
        clone.sel_branch = branch
        return clone

    def step(self):
        for branch_id in range(self.num_branches):
            prop = self.contribute(branch_id)
            self.prop_grid[branch_id][-1] += [prop]

        # Loop party id over
        self.curr_party += 1
        if self.curr_party >= self.num_parties:
            self.curr_party = 0
            self.curr_round += 1
            for branch_id in range(self.num_branches):
                self.prop_grid[branch_id] += [[]]

    def contribute(self, branch: int):
        return "Hello world."

    def _clone(self):
        d = Debate(model=self.model, tokenizer=self.tokenizer)
        for k, v in self.__dict__.items():
            d.__setattr__(k, v)
        return d
