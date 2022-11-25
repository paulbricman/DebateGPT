import string
import pytest
import re
from typing import Any, Dict

from debategpt.training.orchestrator import DebateOrchestrator
from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from trlx.utils.loading import get_model
from trlx.utils import Clock
from transformers import AutoModelForSequenceClassification, AutoTokenizer, ZeroShotClassificationPipeline, pipeline
from accelerate import Accelerator


@pytest.fixture
def orch():
    config = TRLConfig.load_yaml("../configs/debate_ft_config.yml")
    model: AcceleratePPOModel = get_model(config.model.model_type)(config)
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        'cross-encoder/nli-deberta-v3-xsmall')
    nli_tok = AutoTokenizer.from_pretrained(
        'cross-encoder/nli-deberta-v3-xsmall')
    nli_pipe = pipeline(
        "zero-shot-classification",
        model=nli_model,
        tokenizer=nli_tok,
        device=model.accelerator.device)

    orch = DebateOrchestrator(model, nli_pipe)
    return orch


@pytest.fixture
def short_ddc():
    return {
        "num_debates": 2,
        "num_parties": 2,
        "num_rounds": 2,
        "num_facts": 2,
        "objectives": [
            [1, 0],
            [0, 1],
        ]
    }


@pytest.fixture
def long_ddc():
    return {
        "num_debates": 1,
        "num_parties": 5,
        "num_rounds": 10,
        "num_facts": 5,
        "objectives": [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    }


def test_default_debate_configs(orch: DebateOrchestrator):
    ddcs = orch.default_debate_configs()

    assert len(ddcs) == 2
    assert ddcs[0]["num_parties"] == len(ddcs[0]["objectives"])
    assert ddcs[0]["num_parties"] == len(ddcs[0]["objectives"][0])
    assert ddcs[0]["num_facts"] <= 5


def test_ephemeral_generate(orch: DebateOrchestrator):
    prompts = ["", "Hi", "This is the last sentence of this document.\n"]
    experience = orch.ephemeral_generate(prompts)

    assert len(experience["texts"]) == len(prompts)
    assert experience["response_tensors"].size(0) == len(prompts)
    assert experience["all_rewards"].size(1) > 1
    assert experience["all_rewards"][0][-1].tolist() == 0.0, "Early KL somehow non-zero"


def test_long_ephemeral_generate(orch: DebateOrchestrator):
    prompts = ["This is a very long text. " * 200]
    experience = orch.ephemeral_generate(prompts)

    assert len(experience["texts"]) == len(prompts)
    assert experience["response_tensors"].size(0) == len(prompts)
    assert experience["all_rewards"].size(1) > 1
    assert experience["all_rewards"][0][-1].tolist() == 0.0, "Early KL somehow non-zero"


def test_generate_headers(orch: DebateOrchestrator, short_ddc: Dict[str, Any]):
    aliases = string.ascii_uppercase[:short_ddc["num_parties"]]
    headers, facts = orch.create_headers(short_ddc, aliases)

    assert len(headers) == 2
    assert len(headers[0]) > 10
    assert facts[0][0] != facts[0][1], "Facts seem identical, generation is off"


def test_rollout_debate(orch: DebateOrchestrator,
                        short_ddc: Dict[str, Any], long_ddc: Dict[str, Any]):
    clock = Clock()
    experiences, facts, texts, clock = orch.rollout_debate(short_ddc, clock)
    experience = experiences[0][0]

    assert experience["response_tensors"].size(0) == short_ddc["num_debates"]
    assert experience["all_rewards"].size(1) > 1
    assert experience["all_rewards"][0][-1].tolist() == 0.0, "Early KL somehow non-zero"

    assert len(experiences) == short_ddc["num_rounds"]
    assert len(experiences[0]) == short_ddc["num_parties"]

    print(texts)

    experiences, facts, texts, clock = orch.rollout_debate(long_ddc, clock)
    experience = experiences[0][0]

    assert experience["response_tensors"].size(0) == long_ddc["num_debates"]
    assert experience["all_rewards"].size(1) > 1
    assert experience["all_rewards"][0][-1].tolist() == 0.0, "Early KL somehow non-zero"

    assert len(experiences) == long_ddc["num_rounds"]
    assert len(experiences[0]) == long_ddc["num_parties"]

    print(texts)
    assert False
