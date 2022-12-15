from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import string
from copy import deepcopy
from typing import Union, Tuple, List
import networkx as nx
import torch

class Debate:
    def __init__(
            self,
            num_parties=2,
            objectives=None,
            model="distilgpt2",
            tokenizer=None,
            nli_pipe=None
    ):
        """
        Main Debate object used to run parallel debates, select propositions out of them (along party, round, and branch dimensions), etc.
        """
        self.num_parties = num_parties

        if objectives:
            self.objectives = objectives
        else:
            self.objectives = [[1, 0], [0, 1]]

        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(model)

        self.tokenizer = tokenizer
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif not tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(model)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.nli_pipe = nli_pipe
        if not nli_pipe:
            self.nli_pipe = pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-deberta-v3-xsmall")

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

    def flattened_props(self):
        """
        Same as self.props(), but with objects flattened in a single list.
        """
        party_idx, round_idx, branch_idx  = self._sel_idx()
        props = []

        for branch_id in branch_idx:
            for round_id in round_idx:
                for party_id in party_idx:
                    props += [self.prop_grid[branch_id][round_id][party_id]]

        return props

    def props(self):
        """
        Return an object with the same structure as self.prop_grid, but with the selectors applied.
        """
        party_idx, round_idx, branch_idx  = self._sel_idx()
        props = []

        for branch_id in branch_idx:
            branch = []
            for round_id in round_idx:
                round = []
                for party_id in party_idx:
                    round += [self.prop_grid[branch_id][round_id][party_id]]
                branch += [round]
            props += [branch]

        return props

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
            props = self._contribute()
            for branch_id in range(self.num_branches):
                self.prop_grid[branch_id][-1] += [props[branch_id]]

            self.curr_party += 1
            if self.curr_party >= self.num_parties:
                self.curr_party = 0
                self.curr_round += 1
                for branch_id in range(self.num_branches):
                    self.prop_grid[branch_id] += [[]]

    def transcript(self):
        """
        Generate debate transcript of selection for human debugging.
        """
        party_idx, round_idx, branch_idx = self._sel_idx()

        transcript = ""
        for branch_id in branch_idx:
            branch_transcript = ""
            for round_id in round_idx:
                for party_id in party_idx:
                    if round_id < self.curr_round or party_id < self.curr_party:
                        branch_transcript += f"{self.aliases[party_id]}: {self.prop_grid[branch_id][round_id][party_id]}\n"

            transcript += f"\nBranch #{branch_id}\n\n{branch_transcript}---"

        return transcript

    def render(self):
        party_idx, round_idx, branch_idx = self._sel_idx()
        obj_header = f"The table below denotes the allegiances established among the parties which took part in the debate. For instance, a high value at location (A, B) indicates that A supported B.\n\nx"
        for target_id in party_idx:
            obj_header += f"\t{self.aliases[target_id]}"

        for source_id in party_idx:
            obj_header += f"\n{self.aliases[source_id]}"
            for target_id in party_idx:
                obj_header += f"\t{self.objectives[source_id][target_id]}"

        prompts = [obj_header] * len(branch_idx)
        for branch_id in branch_idx:
            if len(self.facts[branch_id]) > 0:
                prompts[branch_id] += f"\n\nThe list below denotes facts which have been deemed established for the purpose of the debate.\n\n"
                for fact in self.facts[branch_id]:
                    prompts[branch_id] += f"- {fact}\n"
            else:
                prompts[branch_id] += "\n"

            prompts[branch_id] += "\nThe rest of this document contains a transcript of the debate in the context of the facts listed above, each brief utterance being one sentence long. This is what the parties said:\n\n"
            for round_id in round_idx:
                for party_id in party_idx:
                    if round_id < self.curr_round or party_id < self.curr_party:
                        prompts[branch_id] += f"{self.aliases[party_id]}: {self.prop_grid[branch_id][round_id][party_id]}\n"

        return prompts

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
        """
        Return argument graph for each branch.
        """
        party_idx, round_idx, branch_idx = self._sel_idx()
        Gs = []
        for branch_id in branch_idx:
            G = nx.DiGraph()
            num_props = len(round_idx) * len(party_idx)
            num_items = num_props + len(self.facts[branch_id])
            G.add_nodes_from(range(num_items))

            item_type = {}
            item_content = {}
            item_party = {}
            item_round = {}

            for round_id in round_idx:
                for party_id in party_idx:
                    node_id = round_id * len(party_idx) + party_id

                    item_type[node_id] = "contribution"
                    item_content[node_id] = self.prop_grid[branch_id][round_id][party_id]
                    item_party[node_id] = party_id
                    item_round[node_id] = round_id

            for fact_id, fact in enumerate(self.facts[branch_id]):
                item_type[fact_id + num_props] = "fact"
                item_content[fact_id + num_props] = fact
                item_party[fact_id + num_props] = -1
                item_round[fact_id + num_props] = -1

            nx.set_node_attributes(G, item_type, "type")
            nx.set_node_attributes(G, item_content, "content")
            nx.set_node_attributes(G, item_party, "party")
            nx.set_node_attributes(G, item_round, "round")

            weighted_edges = []
            run_items = self.branch(branch_id).flattened_props() + self.facts[branch_id]
            run_items = [e if e != "" else "?" for e in run_items]
            run_scores = self.nli_pipe(
                run_items,
                run_items,
                multi_label=True,
                hypothesis_template="{}")

            run_weights = []
            for outbound_id in range(num_items):
                for inbound_id in range(num_items):
                    if outbound_id != inbound_id and inbound_id < num_props:
                        ref_in_id = run_scores[outbound_id]["labels"].index(run_items[inbound_id])
                        run_weights += [(outbound_id, inbound_id,
                                         round(run_scores[outbound_id]["scores"][ref_in_id], 2))]

            G.add_weighted_edges_from(run_weights)
            pageranks = nx.pagerank(G)
            nx.set_node_attributes(G, pageranks, "score")
            Gs += [G]

        return Gs

    def prefix_allow_tokens(self):
        def func(batch_id: int, input_ids: torch.Tensor) -> List[int]:
            last_tok = input_ids.tolist()[-1]
            if last_tok in [13, 30, 0]:
                return [50256]
            return list(range(50255))
        return func

    def _contribute(self):
        """
        Generate a new contribution across branches.
        """
        max_new_toks = 100
        prompts = self.render()
        batch = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=self.model.config.n_ctx - max_new_toks)

        samples = self.model.generate(
            batch["input_ids"],
            bad_words_ids=[[198], [628]],
            do_sample=True,
            top_p=0.9,
            top_k=40,
            no_repeat_ngram_size=2,
            prefix_allowed_tokens_fn=self.prefix_allow_tokens(),
            max_length=self.model.config.n_ctx,
            exponential_decay_length_penalty=(20, 0.9),
            renormalize_logits=True,
        )

        query_tensors = batch.input_ids
        response_tensors = samples[:, query_tensors.shape[1]:]
        props = self.tokenizer.batch_decode(response_tensors,
                                                     skip_special_tokens=True)

        return props

    def _clone(self):
        """
        Creates a mostly-deep copy of the current Debate object. The more heavy-weight models and the associated tokenizers are shallow-copied.
        """
        d = Debate(model=self.model, tokenizer=self.tokenizer, nli_pipe=self.nli_pipe)
        for k, v in self.__dict__.items():
            d.__setattr__(k, v)
        return d

    def _sel_idx(self):
        """
        Return lists of indices based on current selection (as opposed to having a range).
        """
        party_idx = self.sel_party
        if self.sel_party == None:
            party_idx = list(range(self.num_parties))
        elif isinstance(self.sel_party, int):
            party_idx = [party_idx]

        round_idx = self.sel_round
        if self.sel_round == None:
            round_idx = list(range(self.curr_round))
        elif isinstance(self.sel_round, tuple):
            round_idx = list(range(*self.sel_round))
        elif isinstance(self.sel_round, int):
            round_idx = [round_idx]

        branch_idx = self.sel_branch
        if self.sel_branch == None:
            branch_idx = list(range(self.num_branches))
        elif isinstance(self.sel_branch, int):
            branch_idx = [branch_idx]

        return party_idx, round_idx, branch_idx


def distance(d1: Union[Debate, str], d2: Union[Debate, str]):
    """
    Returns an estimate of the ideological distance between two selections of propositions.
    """
    assert isinstance(d1, (Debate, str)) and isinstance(
        d2, (Debate, str)), "Distance can only be computed between objects which are either Debate objects or str."

    if isinstance(d1, str):
        props1 = [d1]
    else:
        props1 = d1.flattened_props()

    if isinstance(d2, str):
        props2 = [d2]
    else:
        props2 = d2.flattened_props()

    nli_pipe = None
    if isinstance(d1, Debate):
        nli_pipe = d1.nli_pipe
    elif isinstance(d2, Debate):
        nli_pipe = d2.nli_pipe
    else:
        nli_pipe = pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-deberta-v3-xsmall")

    scores1 = nli_pipe(props1, props2, multi_label=True, hypothesis_template="{}")
    scores2 = nli_pipe(props2, props1, multi_label=True, hypothesis_template="{}")
    scores = [e["scores"] for e in scores1] + [e["scores"] for e in scores2]
    scores = [e for f in scores for e in f]
    score = sum(scores) / len(scores)

    return score
