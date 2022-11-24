from transformers import AutoModel, AutoTokenizer
import string
from copy import deepcopy
from typing import Union, Tuple, List


class Debate:
    def __init__(
            self,
            num_parties=2,
            objectives=None,
            model="distilgpt2",
            tokenizer=None):
        """
        Main Debate object used to run parallel debates, select propositions out of them (along party, round, and branch dimensions), etc.
        """
        self.num_parties = num_parties

        if objectives:
            self.objectives = objectives
        else:
            self.objectives = [[1, 0], [0, 1]]

        if isinstance(model, str):
            self.model = AutoModel.from_pretrained(model)

        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model)

        self.curr_party = 0
        self.curr_round = 0
        self.num_branches = 1
        self.sel_party = None
        self.sel_round = None
        self.sel_branch = None
        self.aliases = string.ascii_uppercase[:num_parties]
        self.prop_grid = [[[]]]  # branch x round x party contribution
        self.facts = [[]]  # branch x facts

    def party(self, party_id: Union[int, List[int], type(None)]):
        """
        Selects party for subsequent operations (e.g. distance, transcript). Does NOT mutate in-place.
        """
        assert isinstance(party_id, (int, list, type(
            None))), "Party selector should be either an int (i.e. one party id), a list of ints (i.e. multiple party ids), or None to deselect."
        if isinstance(party_id, int):
            assert party_id < self.num_parties and party_id >= 0, f"Current debate only has {self.num_parties} (zero-indexed) parties. You asked for party {party_id}, which is unavailable."
        elif isinstance(party_id, list):
            for party_idx in party_id:
                assert party_idx < self.num_parties and party_idx >= 0, f"Current debate only has {self.num_parties} (zero-indexed) parties. You asked for party {party_idx}, which is unavailable."

        clone = self._clone()
        clone.sel_party = party_id
        return clone

    def round(self, round_id: Union[int, type(None)],
              round_end: Union[int, type(None)] = None):
        """
        Selects round for subsequent operations (e.g. distance, transcript). Does NOT mutate in-place.
        """
        assert isinstance(round_id, (int, type(None))) and isinstance(round_end, (int, type(
            None))), "Round selector requires either an int (i.e. one round id), a pair of two ints (i.e. from the first to the second, not included), or None to deselect."
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
        """
        Selects branch for subsequent operations (e.g. distance, transcript). Does NOT mutate in-place.
        """
        assert isinstance(branch_id, (int, list, type(
            None))), "Branch selector should be either an int (i.e. one branch id) or a list of ints (i.e. multiple branch ids)."
        if isinstance(branch_id, int):
            assert branch_id < self.num_parties and branch_id >= 0, f"Current debate only has {self.num_branches} (zero-indexed) branches. You asked for branch {branch_id}, which is unavailable."
        elif isinstance(branch_id, list):
            for branch_idx in branch_id:
                assert branch_idx < self.num_parties and branch_idx >= 0, f"Current debate only has {self.num_branches} (zero-indexed) branches. You asked for branch {branch_idx}, which is unavailable."

        clone = self._clone()
        clone.sel_branch = branch_id
        return clone

    def selection(self):
        """
        Returns a spec of the selection associated with the current object.
        """
        return {
            "party": self.sel_party,
            "round": self.sel_round,
            "branch": self.sel_branch,
        }

    def play(self, num_rounds: int = 1):
        """
        Runs the debate(s) for `num_rounds` rounds. If not on fresh new round, then the result will have the same current party. Mutates in-place. Parallel debates are advanced in sync.
        """
        for round_id in range(num_rounds):
            for party_id in range(self.num_parties):
                self.step()

    def step(self, num_steps: int = 1):
        """
        Runs the debate(s) for `num_steps` individual steps, meaning that one should expect this many propositions being contributed to each parallel branch. Mutates in-place. Parallel debates are advanced in sync.
        """
        for step_id in range(num_steps):
            for branch_id in range(self.num_branches):
                prop = self.contribute(branch_id)
                self.prop_grid[branch_id][-1] += [prop]

            self.curr_party += 1
            if self.curr_party >= self.num_parties:
                self.curr_party = 0
                self.curr_round += 1
                for branch_id in range(self.num_branches):
                    self.prop_grid[branch_id] += [[]]

    def fork(self, forking_factor: int = 2):
        """
        Forks the current debate(s) into `forking_factor` copies. You can fork multiple times in a row. Functions for advancing the debate(s) map out to all parallel branches. Mutates in-place.
        """
        self.prop_grid *= forking_factor
        self.prop_grid = [deepcopy(e) for e in self.prop_grid]
        self.facts *= forking_factor
        self.facts = [deepcopy(e) for e in self.facts]
        self.num_branches *= forking_factor

    def establish(self, facts: Union[str, List[str]], branch: int = None):
        """
        Establishes the given facts in the target branch. Target to `None` branch to establish the same facts across all available parallel debates. Mutates in-place.
        """
        if isinstance(facts, str):
            facts = [facts]
        if not branch:
            for branch_id in range(self.num_branches):
                self.facts[branch_id] += facts
                print(
                    "fail should pass through this",
                    branch_id,
                    branch,
                    self.num_branches,
                    self.facts[branch_id])
        else:
            self.facts[branch] += facts

    def graph(self):
        return None

    def contribute(self, branch: int):
        return "Hello world."

    def _clone(self):
        """
        Creates a mostly-deep copy of the current Debate object. The more heavy-weight models and the associated tokenizers are shallow-copied.
        """
        d = Debate(model=self.model, tokenizer=self.tokenizer)
        for k, v in self.__dict__.items():
            d.__setattr__(k, v)
        return d


def distance(d1: Union[Debate, str], d2: Union[Debate, str]):
    """
    Returns an estimate of the ideological distance between two selections of propositions.
    """
    assert isinstance(d1, (Debate, str)) and isinstance(
        d2, (Debate, str)), "Distance can only be computed between objects which are either Debate objects or str."
    return 0.42
