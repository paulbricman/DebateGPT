"""
A custom version of [trlx.py](https://github.com/CarperAI/trlx/blob/master/trlx/trlx.py) adapted to make use of `DebateOrchestrator` in an online fashion.
"""


from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from trlx.utils.loading import get_model
from trlx.pipeline.offline_pipeline import PromptPipeline
from src.orchestrator import DebateOrchestrator
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def train() -> AcceleratePPOModel:
    """
    Dispatches debate fine-tuning in an online fashion through the custom orchestrator.
    """
    config = TRLConfig.load_yaml("configs/debate_ft_config.yml")
    model: AcceleratePPOModel = get_model(config.model.model_type)(config)

    nli_model = AutoModelForSequenceClassification.from_pretrained(
        'cross-encoder/nli-deberta-v3-xsmall')
    nli_tok = AutoTokenizer.from_pretrained(
        'cross-encoder/nli-deberta-v3-xsmall')
    nli_model = model.accelerator.prepare(nli_model)

    orch = DebateOrchestrator(model, nli_model, nli_tok)

    # Two lines below are just to play nice with trlx
    eval_pipeline = PromptPipeline([" "] * 2, model.tokenizer)
    model.add_eval_pipeline(eval_pipeline)

    orch.make_experience(config.method.num_rollouts)
    model.learn()
    return model
