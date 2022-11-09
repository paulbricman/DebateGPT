from typing import Callable, Dict, Any

import torch
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.ppo_types import PPORLElement
from trlx.model import BaseRLModel
from trlx.model.nn.ppo_models import GPTHeadWithValueModel, GPTHydraHeadWithValueModel
from trlx.orchestrator import Orchestrator, register_orchestrator
from trlx.pipeline import BasePipeline
from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.utils import Clock
from trlx.utils.modeling import logprobs_from_logits


@register_orchestrator
class DebateOrchestrator(Orchestrator):
    """
    Orchestrator generates debate experience, packages them up in PPORLElements, and pushes them to the store.
    """

    def __init__(
        self,
        model: BaseRLModel,
        reward_fn: Callable,
        metric_fn: Callable = None,
    ):
        self.rl_model = model

        if not hasattr(self.rl_model.model, "frozen_head"):
            self.ref_model = self.rl_model.get_arch(self.rl_model.config)

        self.rl_model.orch = self
        self.rl_model.reward_fn = reward_fn
        self.rl_model.metric_fn = metric_fn


    def make_experience(self, debate_config: Dict[str, Any], num_debates: int = 1024, iter_count: int = 0):
        """
        Generates `num_debates` debates between `num_parties` parties for `num_rounds` rounds. Computes rewards for each proposition, packages each proposition experience as a separate PPORLElement, and finally pushes all of them to the store.
        """
        ppo_rl_elements = []
        stats = {}
        clock = Clock()

        # TODO: Handle rounds (tick!), generate `experiences`
        # TODO: Tack score on final token of each experience

        for round in debate_config["num_rounds"]:
            es = experiences[round]
            new_ppo_rl_elements = [
                PPORLElement(
                    query_tensor=es["query_tensors"][i, :],
                    response_tensor=es["response_tensors"][i, :],
                    logprobs=es["all_logprobs"][i, :],
                    values=es["all_values"][i, :],
                    rewards=es["all_rewards"][i, :],
                )
                for i in range(es["query_tensors"].size()[0])
            ]

        ppo_rl_elements += new_ppo_rl_elements
        stats = {"exp_time": exp_time}
        self.rl_model.accelerator.log(stats, step=iter_count)
        self.rl_model.push_to_store(ppo_rl_elements)


    def default_debate_config(self):
        return {
            "num_parties": 3,
            "num_rounds": 5,
            "num_facts": 3,
            "objectives": [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]
        }


    def ephemeral_generate(self, prompts):
        # Generate
        ephemeral_pipeline = PromptPipeline(prompts, self.rl_model.tokenizer)
        pipeline_loader = ephemeral_pipeline.create_loader(
            len(prompts), shuffle=True
        )
        pipeline_loader = self.rl_model.accelerator.prepare(pipeline_loader)
        pipeline_iterator = iter(pipeline_loader)
        batch: PromptBatch = next(self.pipeline_iterator)
        samples = self.rl_model.generate(**batch)

        # Wrangle
        query_tensors = batch.input_ids
        response_tensors = samples[:, query_tensors.shape[1] :]
        texts = self.rl_model.tokenizer.batch_decode(
            samples, skip_special_tokens=True
        )
        all_tokens = torch.cat(
            (query_tensors.to(samples.device), response_tensors), dim=1
        )

        # Handle logprobs
        with torch.no_grad():
            logits, _, v = self.rl_model.model(all_tokens)

            if hasattr(self.rl_model.model, "frozen_head"):
                ref_logits = self.rl_model.model.forward_hydra(
                    all_tokens, return_dict=False
                )
            else:
                ref_logits, _, _ = self.ref_model(all_tokens.cpu())

        ref_logits = ref_logits.to(self.rl_model.accelerator.device)
        logprobs = logprobs_from_logits(logits[:, :-1, :], all_tokens[:, 1:])
        ref_logprobs = logprobs_from_logits(
            ref_logits[:, :-1, :], all_tokens[:, 1:]
        )
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
            "all_rewards": all_rewards
        }
