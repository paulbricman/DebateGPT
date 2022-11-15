from typing import List, Any, Dict
import networkx as nx
import pytest

from src.reward import compute_pagerank, compute_mixing, enrich_experiences, compute_arc_weights, compose_graphs
from src.orchestrator import DebateOrchestrator
from trlx.utils import Clock
from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from trlx.utils.loading import get_model


@pytest.fixture
def dummy_graphs():
    G = nx.DiGraph()
    G.add_weighted_edges_from([
        (0, 1, 0.2),
        (1, 0, -0.5),
        (1, 2, 0.1),
        (2, 1, -0.7),
        (0, 2, -0.1),
        (2, 0, 0.4),
        (3, 4, -0.2),
        (4, 3, -0.5),
        (4, 5, -0.3),
        (5, 4, -0.7),
        (3, 5, -0.1),
        (5, 3, 0.1),
        (0, 3, 1.0),
        (1, 4, 0.7),
        (2, 5, 0.9),
    ])
    H = G.copy()
    return [G, H]


@pytest.fixture
def ddc():
    return {
        "num_debates": 2,
        "num_parties": 3,
        "num_rounds": 2,
        "num_facts": 2,
        "objectives": [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    }


@pytest.fixture
def orch():
    config = TRLConfig.load_yaml("../configs/debate_ft_config.yml")
    model: AcceleratePPOModel = get_model(config.model.model_type)(config)
    orch: DebateOrchestrator = DebateOrchestrator(model)
    return orch


@pytest.fixture
def dummy_props():
    # Two rounds, three parties, high internal coherence
    props = [
        "Longtermism is amazing.", "Longtermism is stupid.",
        "Longtermism is some new philosophy.",
        "It is important to care about people living in the long-term future.",
        "It is useless to think of people who are not alive today.",
        "Longtermism explores the idea of taking into account the well-being of future people."
    ]
    return [props, props.copy()]


def test_compute_pagerank(dummy_graphs: List[Any], ddc: Dict[str, Any]):
    pageranks = compute_pagerank(dummy_graphs, ddc)

    assert pageranks[0][0] == pageranks[0][ddc["num_parties"]]
    assert pageranks[0] == pageranks[1]


def test_compute_mixing(dummy_graphs: List[Any], ddc: Dict[str, Any]):
    mixings = compute_mixing(dummy_graphs, ddc)

    assert mixings[0] == mixings[1]
    assert all([e >= -1 and e <= 1 for e in mixings])


def test_enrich_experiences(ddc: Dict[str, Any], orch: DebateOrchestrator,
                            dummy_graphs: List[Any]):
    pageranks = compute_pagerank(dummy_graphs, ddc)
    clock = Clock()
    experiences, facts, texts, clock = orch.rollout_debate(ddc, clock)
    initial_std_reward = experiences[0][0]["all_rewards"][0][0].tolist()
    initial_final_reward = experiences[0][0]["all_rewards"][0][-1].tolist()
    enriched_experiences = enrich_experiences(experiences, pageranks, ddc)

    assert enriched_experiences[0][0]["all_rewards"][0][0].tolist(
    ) == initial_std_reward
    assert enriched_experiences[0][0]["all_rewards"][0][-1].tolist(
    ) == initial_final_reward + pageranks[0][0]


def test_compute_arc_weights(dummy_props: List[List[str]], ddc: Dict[str,
                                                                     Any]):
    weights = compute_arc_weights(dummy_props, ddc)

    assert len(weights) == len(dummy_props)
    assert len(weights[0]) == len(dummy_props[0]) * (len(dummy_props[0]) - 1)
    assert all([e[2] <= 1. and e[2] >= 0. for e in weights[0]])


def test_compute_graphs(dummy_props: List[List[str]], ddc: Dict[str, Any]):
    graphs = compose_graphs(dummy_props, ddc)

    assert len(graphs[0].nodes) == len(dummy_props[0])
    assert len(graphs[0].edges) == len(dummy_props[0]) * (len(dummy_props[0]) - 1)
