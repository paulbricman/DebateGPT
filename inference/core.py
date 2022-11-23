from transformers import AutoModel, AutoTokenizer
import string
from copy import deepcopy
from typing import Union, Tuple


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

    def who(self, party: Union[int, List[int], type(None)]):
        assert isinstance(party, (int, list, type(None))), "Party selector should be either an int (i.e. one party id), a list of ints (i.e. multiple party ids), or None to deselect."
        if isinstance(party, int):
            assert party < self.num_parties and party >= 0, f"Current debate only has {self.num_parties} (zero-indexed) parties. You asked for party {party}, which is unavailable."
        elif isinstance(party, list):
            for party_id in party:
                assert party_id < self.num_parties and party_id >= 0, f"Current debate only has {self.num_parties} (zero-indexed) parties. You asked for party {party_id}, which is unavailable."

        clone = self._clone()
        clone.sel_party = party
        return clone

    def when(self, round: Union[int, Tuple[int, int], type(None)]):
        assert isinstance(round, (int, tuple, type(None))), "Round selector should be either an int (i.e. one round id), a tuple of two ints (i.e. from the first to the second, not included), or None to deselect."
        if isinstance(round, int):
            assert round <= self.curr_round and round >= 0, f"Current debate has only been running for {self.curr_round} (zero-indexed) rounds. You asked for round {round}, which hasn't happened yet."
        elif isinstance(round, tuple):
            assert round[0] <= round[1], "Start round selector should be lower or equal than the end selector."
            assert round[0] <= self.curr_round and round[0] >= 0 and round[1] <= self.curr_round and round[1] >= 0, f"Current debate has only been running for {self.curr_round} (zero-indexed) rounds. You asked for rounds outside this range."

        clone = self._clone()
        clone.sel_round = round
        return clone

    def which(self, branch: Union[int, List[int]]):
        assert isinstance(branch, (int, list, type(None))), "Branch selector should be either an int (i.e. one branch id) or a list of ints (i.e. multiple branch ids)."
        if isinstance(branch, int):
            assert branch < self.num_parties and branch >= 0, f"Current debate only has {self.num_branches} (zero-indexed) branches. You asked for branch {branch}, which is unavailable."
        elif isinstance(branch, list):
            for branch_id in branch:
                assert branch_id < self.num_parties and branch_id >= 0, f"Current debate only has {self.num_branches} (zero-indexed) branches. You asked for branch {branch_id}, which is unavailable."
        clone = self._clone()
        clone.sel_branch = branch
        return clone

    def selection(self):
        return {
            "sel_party": self.sel_party,
            "sel_round": self.sel_round,
            "sel_branch": self.sel_branch
        }

    def round(self):
        for party_id in range(self.num_parties):
            self.step()

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


def dawkins(prop1, prop2):
    assert is_prop(prop1) and is_prop(prop2), "Arguments are not single propositions."
    return 0.42


def overton(prop, pos):
    assert is_prop(prop) and is_position(pos), "Either the first argument is not a single proposition, or the second argument is not a correctly selected position."
    return 0.42


def kuhn(pos1, pos2):
    assert is_position(pos1) and is_position(pos2), "Arguments are not correctly selected positions."
    return 0.42


def is_prop(d):
    return isinstance(d, str) or all([
        isinstance(d.sel_party, int),
        isinstance(d.sel_round, int),
        isinstance(d.sel_branch, int)
    ])


def is_position(d):
    return all([
        isinstance(d.sel_party, int),
        isinstance(d.sel_round, tuple),
        isinstance(d.sel_branch, int)
    ])


def is_branch(d):
return all([
        isinstance(d.sel_party, type(None)),
        isinstance(d.sel_round, tuple),
        isinstance(d.sel_branch, int)
    ])
