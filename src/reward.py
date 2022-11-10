from typing import Dict, Any, List
import networkx as nx


def reward(experiences: List[Dict[str, Any]], debate_config: Dict[str, Any]) -> experiences: List[Dict[str, Any]]:
    """
    Coordinate the enrichment of experiences with rewards. In terms of terminology, reward in this codebase refers to the sum of KL penalties and domain-specific scores. Although the relevant aspects of debate configuration can be inferred from the structure of the `experiences` object, having to include it explicitly makes it clear that they have to share the same structure.
    """
    # Collapse round and party dims into a flattened props dim
    props = []
    for run in experiences:
        round_props = []
        for round in run:
            for party in round:
                round_props += [party["props"]]
        props += [round_props]

    graphs = compose_graphs(props)
    scores = compute_pagerank(graphs)
    enriched_es = enrich_experiences(experiences, scores)
    return enriched_es


def compose_graphs(props: List[List[str]]) -> graphs: List[nx.classes.DiGraph]:
    """
    Compose a weighted directed graph using networkx where nodes represent propositions and arc represent relations of support between them.
    """
    assert all([len(e) == len(props[0]) for e in props]), "Runs differ in num_props!"

    weights = compute_arc_weights(props)
    graphs = []
    for run_id in range(props):
        D = nx.DiGraph()
        D.add_weighted_edges_from(weights[run_id])
        graphs += [D]

    return graphs


def compute_arc_weights(props: List[List[str]]) -> weights: List[List[Tuple[int, int, float]]]:
    """
    Run pairs of props through NLI pipeline to compute arc weights for each graph. The predefined zero-shot text classification is used due to it conveniently wrapping NLI-related logic, although its original goal was to be used in a different application.
    """
    paired_props = []
    for run in props:
        round_paired_props = [(e, f) for e in run for f in run if e != f]
        paired_props += [round_paired_props]

    # NLI
    # zeroshot pipeline?
    # accelerate model registration?
    # inspo from trlx generate wrapper?
    # investigate choice of NLI model a bit?


def compute_pagerank(graphs: List[nx.classes.DiGraph]) -> scores: List[List[float]]:
    """
    Run and wrangle data for PageRank on each graph representing a run.
    """
    pageranks = [nx.pagerank(e) for e in graphs]
    pageranks = [list(e.values()) for e in pageranks]
    return pageranks


def enrich_experiences(experiences: List[Dict[str, Any]], scores: List[List[float]], debate_config: Dict[str, Any]) -> experiences: List[Dict[str, Any]]:
    """
    Tack scores on the final token of each experience.
    """
    for run_id, run in enumerate(experiences):
        for round_id, round in enumerate(run):
            for party_id, party in enumerate(round):
                prop_id = round_id * debate_config["num_parties"] + party
                experiences[run_id][round_id][party_id]["all_rewards"][-1] = scores[run_id][prop_id]

    return experiences
