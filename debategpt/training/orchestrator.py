from typing import Any, Dict, List, Tuple
import string
import random
import wandb
import torch
from trlx.data.ppo_types import PPORLElement
from trlx.model import BaseRLModel
from trlx.orchestrator import Orchestrator
from trlx.utils import Clock
from trlx.utils.modeling import logprobs_from_logits
from debategpt.training.reward import reward
from transformers import ZeroShotClassificationPipeline, StoppingCriteria
import pandas as pd
import torch


class DebateOrchestrator(Orchestrator):
    """
    Orchestrator generates debate experience, packages them up in PPORLElements, and pushes them to the store.
    """

    def __init__(
            self,
            model: BaseRLModel,
            nli_pipe: ZeroShotClassificationPipeline):
        self.rl_model = model
        self.nli_pipe = nli_pipe

        if not hasattr(self.rl_model.model, "frozen_head"):
            self.ref_model = self.rl_model.get_arch(self.rl_model.config)

        self.rl_model.reward_fn = None
        self.rl_model.metric_fn = None

        self.rl_model.orch = self

    def make_experience(self, _: Any = None, iter_count: int = 0):
        clock = Clock()
        debate_configs = self.default_debate_configs()
        for debate_config in debate_configs:
            self.make_experience_type(debate_config, clock, iter_count)

    def make_experience_type(self, debate_config: Dict[str, Any],
                             clock: Clock, iter_count: int):
        """
        Generate debates in parallel following a certain configuration, bundle up the experiences together with associated rewards as PPORLElements, and push them to store.
        """
        ppo_rl_elements = []
        stats = {}

        experiences, facts, texts, clock = self.rollout_debate(debate_config,
                                                               clock)
        experiences, mixings, scores, props = reward(
            experiences, facts, debate_config, self.nli_pipe)

        for round_id in range(debate_config["num_rounds"]):
            for party_id in range(debate_config["num_parties"]):
                es = experiences[round_id][party_id]
                new_ppo_rl_elements = [
                    PPORLElement(
                        query_tensor=es["query_tensors"][i, :],
                        response_tensor=es["response_tensors"][i, :],
                        logprobs=es["all_logprobs"][i, :],
                        values=es["all_values"][i, :],
                        rewards=es["all_rewards"][i, :],
                    ) for i in range(debate_config["num_debates"])
                ]

                ppo_rl_elements += new_ppo_rl_elements

        exp_time = clock.tick()
        stats = {
            "facts": pd.DataFrame(facts),
            "prop_scores": pd.DataFrame(zip(props[0], scores[0])),
            "prop_contexts": pd.DataFrame(zip(experiences[1][1]["prompts"], experiences[1][1]["texts"])),
            "assortative_mixing_avg": sum(mixings) /
            debate_config["num_debates"]
        }
        self.rl_model.accelerator.log(stats, step=iter_count)
        self.rl_model.push_to_store(ppo_rl_elements)

    def default_debate_configs(self) -> List[Dict[str, Any]]:
        """
        Specify a sensible configuration for the debates.

        Returns:
            List of debate configs
        """
        random.seed(0)
        num_debate_config_types = 2
        num_debates = 2
        debate_configs = []

        for id in range(num_debate_config_types):
            num_parties = random.randint(2, 3)
            num_facts = random.randint(1, 4)
            num_rounds = random.randint(5, 8)
            objectives = (torch.normal(torch.zeros(
                (num_parties,
                 num_parties)), torch.ones(
                     (num_parties, num_parties))) * 0.25 +
                torch.eye(num_parties)).tolist()
            objectives = [[round(e, 2) for e in f] for f in objectives]

            debate_configs += [{
                "num_debates": num_debates,
                "num_parties": num_parties,
                "num_rounds": num_rounds,
                "num_facts": num_facts,
                "objectives": objectives
            }]

        return debate_configs

    def prefix_allow_tokens(self):
        def func(batch_id: int, input_ids: torch.Tensor) -> List[int]:
            last_tok = input_ids.tolist()[-1]
            if last_tok in [13, 30, 0]:
                return [50256]
            return list(range(50255))
        return func

    def ephemeral_generate(self,
                           prompts: List[str]) -> Dict[str, Any]:
        """
        Utility function which handles one step of generating rollout from prompts in parallel, including tracking logprobs and KLs. For the most part lifted from the source code of PPOOrchestrator.

        Returns:
            An experience
        """
        max_new_toks = 100
        self.rl_model.tokenizer.pad_token = self.rl_model.tokenizer.eos_token
        self.rl_model.tokenizer.padding_side = "left"
        batch = self.rl_model.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=self.rl_model.config.train.seq_length - max_new_toks)

        success = False
        while not success:
            try:
                samples = self.rl_model.generate(
                    **batch,
                    bad_words_ids=[[198], [628]],
                    do_sample=True,
                    top_p=0.9,
                    top_k=40,
                    no_repeat_ngram_size=2,
                    prefix_allowed_tokens_fn=self.prefix_allow_tokens(),
                    max_new_tokens=max_new_toks,
                    renormalize_logits=True)
                success = True
            except:
                pass

        # Wrangle
        query_tensors = batch.input_ids
        response_tensors = samples[:, query_tensors.shape[1]:]
        texts = self.rl_model.tokenizer.batch_decode(response_tensors,
                                                     skip_special_tokens=True)

        all_tokens, attention_mask, position_ids = self.rl_model.get_model_inputs(
            query_tensors.to(response_tensors.device), response_tensors
        )
        with torch.no_grad():
            logits, *_, v = self.rl_model.model(
                all_tokens, attention_mask=attention_mask, position_ids=position_ids
            )
            if hasattr(self.rl_model.model, "frozen_head"):
                ref_logits = self.rl_model.model.forward_hydra(
                    all_tokens,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    return_dict=False,
                )
            else:
                ref_logits, _, *_ = self.ref_model(
                    all_tokens.cpu(),
                    attention_mask=attention_mask.cpu(),
                    position_ids=position_ids.cpu(),
                )

        ref_logits = ref_logits.to(self.rl_model.accelerator.device)
        logprobs = logprobs_from_logits(logits[:, :-1, :], all_tokens[:, 1:])
        ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :],
                                            all_tokens[:, 1:])
        start = query_tensors.size()[1] - 1
        end = query_tensors.size()[1] + response_tensors.size()[1] - 1
        all_values = v[:, start:end]
        all_logprobs = logprobs[:, start:end]
        all_ref_logprobs = ref_logprobs[:, start:end]

        # Handle KL
        kls = all_logprobs - all_ref_logprobs
        non_score_rewards = -self.rl_model.kl_ctl.value * kls
        all_rewards = non_score_rewards.clone()

        # Move to host
        query_tensors = query_tensors.cpu()
        response_tensors = response_tensors.cpu()
        all_logprobs = all_logprobs.cpu()
        all_values = all_values.cpu()
        all_rewards = all_rewards.cpu()

        return {
            "query_tensors": query_tensors,
            "response_tensors": response_tensors,
            "all_logprobs": all_logprobs,
            "all_values": all_values,
            "all_rewards": all_rewards,
            "prompts": prompts,
            "texts": texts,
            "scores": [],
        }

    def rollout_debate(self,
                       debate_config: Dict[str,
                                           Any],
                       clock: Clock) -> Tuple[List[List[Dict[str,
                                                             Any]]],
                                              List[List[str]],
                                              List[str],
                                              Clock]:
        """
        Systematically generate propositions contributed by alternate parties for a number of rounds while keeping track of everything (e.g. logprobs, KLs, tokens, etc.).

        Returns:
            List of lists of experiences (round x party)
            List of lists of facts (run)
            List of debate transcripts
            Clock
        """
        aliases = string.ascii_uppercase[:debate_config["num_parties"]]
        texts, facts = self.create_headers(debate_config, aliases)
        experiences = []

        for round in range(debate_config["num_rounds"]):
            round_experiences = []
            for party in range(debate_config["num_parties"]):
                texts = [e + f"{aliases[party]}:" for e in texts]
                completions = self.ephemeral_generate(texts)
                round_experiences += [completions]
                texts = [
                    e + f + "\n" for e, f in zip(texts, completions["texts"])
                ]

            experiences += [round_experiences]

        return experiences, facts, texts, clock

    def create_headers(
            self, debate_config: Dict[str, Any],
            aliases: List[str]) -> Tuple[List[str], List[List[str]]]:
        """
        Generate (partly procedurally) headers prepended to the actual debate content.

        Returns:
            List of headers (run)
            List of lists of facts (run)
        """
        party_idx = list(range(debate_config["num_parties"]))
        branch_idx = list(range(debate_config["num_debates"]))
        objectives = debate_config["objectives"]

        obj_header = f"The table below denotes the allegiances established among the parties which took part in the debate. For instance, a high value at location (A, B) indicates that A supported B.\n\nx"
        for target_id in party_idx:
            obj_header += f"\t{aliases[target_id]}"

        for source_id in party_idx:
            obj_header += f"\n{aliases[source_id]}"
            for target_id in party_idx:
                obj_header += f"\t{objectives[source_id][target_id]}"

        prompts = [obj_header] * len(branch_idx)

        # Each debate runs with unique facts
        fact_prompt = "The following is a list of concise established facts about the world, one single sentence each. They span history, literature, mathematics, and many other diverse fields:\n\n-"
        fact_prompts = [
            fact_prompt
        ] * debate_config["num_facts"] * debate_config["num_debates"]
        facts = self.ephemeral_generate(fact_prompts)["texts"]
        facts = [e.split("\n")[0].strip() for e in facts]

        fact_headers = [
            facts[e * debate_config["num_facts"]:(e + 1) *
                  debate_config["num_facts"]]
            for e in range(debate_config["num_debates"])
        ]
        raw_facts = fact_headers

        for branch_id in branch_idx:
            if len(fact_headers[branch_id]) > 0:
                prompts[branch_id] += f"\n\nThe list below denotes facts which have been deemed established for the purpose of the debate.\n\n"
                for fact in fact_headers[branch_id]:
                    prompts[branch_id] += f"- {fact}\n"
            else:
                prompts[branch_id] += "\n"

            prompts[branch_id] += "\nThe rest of this document contains a transcript of the debate in the context of the facts listed above, each brief utterance being one sentence long. This is what the parties said:\n\n"

        return prompts, raw_facts

