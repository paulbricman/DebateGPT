from src.orchestrator.debate_orchestrator import DebateOrchestrator
from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from trlx.data.configs import TRLConfig
from trlx.utils.loading import get_model


def test_ephemeral_generate():
    config = TRLConfig.load_yaml("ppo_config.yml")
    model: AcceleratePPOModel = get_model(config.model.model_type)(config)
    orch: DebateOrchestrator = DebateOrchestrator(
            model, reward_fn=reward_fn
        )
    prompts = ["Hello", "Hi"]
    experiences = orch.ephemeral_generate(prompts)
    print(prompts)
    assert False

