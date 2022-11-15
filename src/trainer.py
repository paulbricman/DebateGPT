"""
A custom version of [trlx.py](https://github.com/CarperAI/trlx/blob/master/trlx/trlx.py) adapted to make use of `DebateOrchestrator` in an online fashion.
"""


from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from trlx.utils.loading import get_model
from src.orchestrator import DebateOrchestrator


def train() -> AcceleratePPOModel:
    """
    Dispatches debate fine-tuning in an online fashion through the custom orchestrator.
    """
    config = TRLConfig.load_yaml("configs/debate_ft_config.yml")
    model: AcceleratePPOModel = get_model(config.model.model_type)(config)
    orch = DebateOrchestrator(model)

    orch.make_experience(config.method.num_rollouts)
    model.learn()
    return model
