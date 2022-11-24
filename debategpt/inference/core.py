from transformers import AutoModel, AutoTokenizer
import string
from copy import deepcopy
from typing import Union, Tuple, List


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
        self.prop_grid = [[[]]] # branch x round x party contribution
        self.facts = [[]] # branch x facts

    def party(self, party_id: Union[int, List[int], type(None)]):
        assert isinstance(party_id, (int, list, type(None))), "Party selector should be either an int (i.e. one party id), a list of ints (i.e. multiple party ids), or None to deselect."
        if isinstance(party_id, int):
            assert party_id < self.num_parties and party_id >= 0, f"Current debate only has {self.num_parties} (zero-indexed) parties. You asked for party {party_id}, which is unavailable."
        elif isinstance(party_id, list):
            for party_idx in party_id:
                assert party_idx < self.num_parties and party_idx >= 0, f"Current debate only has {self.num_parties} (zero-indexed) parties. You asked for party {party_idx}, which is unavailable."

        clone = self._clone()
        clone.sel_party = party_id
        return clone

    def round(self, round_id: Union[int, type(None)], round_end: Union[int, type(None)] = None):
        assert isinstance(round_id, (int, type(None))) and isinstance(round_end, (int, type(None))), "Round selector requires either an int (i.e. one round id), a pair of two ints (i.e. from the first to the second, not included), or None to deselect."
        if isinstance(round_id, int):
            assert round_id <= self.curr_round and round_id >= 0, f"Current debate has only been running for {self.curr_round} (zero-indexed) rounds. You asked for round {round_id}, which hasn't happened yet."
        if isinstance(round_end, int):
            assert round_end <= self.curr_round and round_end >= 0, f"Current debate has only been running for {self.curr_round} (zero-indexed) rounds. You asked for round {round_end}, which hasn't happened yet."
            assert round_id <= round_end, "Start round selector should be lower or equal than the end selector."

        clone = self._clone()
        if round_end:
            clone.sel_round = round_id, round_end
        else:
            clone.sel_round = round_id
        return clone

    def branch(self, branch_id: Union[int, List[int]]):
        assert isinstance(branch_id, (int, list, type(None))), "Branch selector should be either an int (i.e. one branch id) or a list of ints (i.e. multiple branch ids)."
        if isinstance(branch_id, int):
            assert branch_id < self.num_parties and branch_id >= 0, f"Current debate only has {self.num_branches} (zero-indexed) branches. You asked for branch {branch_id}, which is unavailable."
        elif isinstance(branch_id, list):
            for branch_idx in branch_id:
                assert branch_idx < self.num_parties and branch_idx >= 0, f"Current debate only has {self.num_branches} (zero-indexed) branches. You asked for branch {branch_idx}, which is unavailable."

        clone = self._clone()
        clone.sel_branch = branch_id
        return clone

    def selection(self):
        return {
            "party": self.sel_party,
            "round": self.sel_round,
            "branch": self.sel_branch,
        }

    def play(self, num_rounds: int = 1):
        for round_id in range(num_rounds):
            for party_id in range(self.num_parties):
                self.step()

    def step(self):
        for branch_id in range(self.num_branches):
            prop = self.contribute(branch_id)
            self.prop_grid[branch_id][-1] += [prop]

        self.curr_party += 1
        if self.curr_party >= self.num_parties:
            self.curr_party = 0
            self.curr_round += 1
            for branch_id in range(self.num_branches):
                self.prop_grid[branch_id] += [[]]

    def inject(self, prop: str):
        assert is_prop(self), "When injecting, you must finely select the injection site (i.e. single party, single round, single branch)."
        self.prop_grid[self.sel_branch][self.sel_round][self.sel_party] = prop

    def fork(self, forking_factor: int = 2):
        self.prop_grid *= forking_factor
        self.num_branches *= forking_factor

    def establish(self, facts: Union[str, List[str]], branch: int):
        if isinstance(facts, str):
            facts = [facts]
        self.facts[branch] += facts

    def graph(self):
        assert is_branch(self), "In order to get convert a branch across several rounds to the graph representation, you must select one accordingly."
        return None

    def contribute(self, branch: int):
        return "Hello world."

    def _clone(self):
        d = Debate(model=self.model, tokenizer=self.tokenizer)
        for k, v in self.__dict__.items():
            d.__setattr__(k, v)
        return d


def distance(d1: Union[Debate, str], d2: Union[Debate, str]):
    assert isinstance(d1, (Debate, str)) and isinstance(d2, (Debate, str)), "Distance can only be computed between objects which are either Debate objects or str."
    return 0.42
