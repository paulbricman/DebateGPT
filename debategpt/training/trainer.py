"""
A custom version of [trlx.py](https://github.com/CarperAI/trlx/blob/master/trlx/trlx.py) adapted to make use of `DebateOrchestrator` in an online fashion.
"""


from trlx.data.configs import TRLConfig
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
from trlx.utils.loading import get_trainer, get_pipeline
from debategpt.training.orchestrator import DebateOrchestrator
from transformers import pipeline
from accelerate import Accelerator


def train() -> AcceleratePPOTrainer:
    """
    Dispatches debate fine-tuning in an online fashion through the custom orchestrator.
    """
    config = TRLConfig.load_yaml("configs/debate_ft_config.yml")
    trainer: AcceleratePPOTrainer = get_trainer(config.train.trainer)(config)

    nli_pipe = pipeline(
        "zero-shot-classification",
        model="cross-encoder/nli-deberta-v3-xsmall",
        device=trainer.accelerator.device)

    orch = DebateOrchestrator(trainer, nli_pipe)

    # Two lines below are just to play nice with trlx
    trlx_pipeline = get_pipeline(config.train.pipeline)(
        [" "] * 2, 1, trainer.tokenizer
    )
    trainer.add_eval_pipeline(trlx_pipeline)

    orch.make_experience(config.method.num_rollouts)
    trainer.learn()
    return trainer
