from typing import Dict, Any, List
import networkx as nx
from accelerate import Accelerator
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def reward(experiences: List[Dict[str, Any]],
           debate_config: Dict[str, Any]) -> List[Dict[str, Any]]:
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


def compose_graphs(props: List[List[str]]) -> List[nx.classes.DiGraph]:
    """
    Compose a weighted directed graph using networkx where nodes represent propositions and arc represent relations of support between them.
    """
    assert all([len(e) == len(props[0])
                for e in props]), "Runs differ in num_props!"

    weights = compute_arc_weights(props)
    graphs = []
    for run_id in range(props):
        D = nx.DiGraph()
        D.add_weighted_edges_from(weights[run_id])
        graphs += [D]

    return graphs


def compute_arc_weights(
        props: List[List[str]],
        debate_config: Dict[str, Any]) -> List[List[Tuple[int, int, float]]]:
    """
    Run pairs of props through NLI pipeline to compute arc weights for each graph. The predefined zero-shot text classification is used due to it conveniently wrapping NLI-related logic , although its original goal was to be used in a different application.
    """
    paired_props = []
    for run in props:
        paired_props += [(e, f) for e in run for f in run if e != f]

    sequences = [e[0] for e in paired_props]
    candidates = [[e[1]] for e in paired_props]

    # Parallelization approach uses models prepared by accelerate. Alternatively, create one pipeline per device and handle dispatch manually. Alternatively, scrape all of that and just fallback to a single pipeline on one GPU, as it would be simple despite not efficient.
    model = AutoModelForSequenceClassification.from_pretrained(
        'cross-encoder/nli-deberta-v3-base')
    tokenizer = AutoTokenizer.from_pretrained(
        'cross-encoder/nli-deberta-v3-base')
    model, optimizer = accelerator.prepare(model)
    nli_pipe = pipeline("zero-shot-classification", model=model)
    scores = nli_pipe(sequences,
                      candidates,
                      multi_label=True,
                      hypothesis_template="{}")["scores"]

    weighted_edges = []
    num_props_per_run = debate_config["num_rounds"] * debate_config[
        "num_parties"]
    num_pairs_per_run = num_props_per_run * (num_props_per_run - 1
                                             )  # No node self-ref.
    for run_id in range(debate_config["num_debates"]):
        run_weighted_edges = []
        run_start = run_id * num_pairs_per_run
        run_end = run_start + num_pairs_per_run
        for prop_id in range(num_props_per_run):
            for pair_id in range(number_props_per_run - 1):
                # Skip self-ref arc.
                effective_target_id = pair_id
                if effective_target >= prop_id:
                    effective_target += 1

                pair_abs_id = run_start + prop_id * (num_props_per_run -
                                                     1) + pair_id
                run_weighted_edges += [(prop_id, effective_target_id,
                                        scores[pair_abs_id])]
        weighted_edges += [run_weighted_edges]

    return weighted_edges


def compute_pagerank(graphs: List[nx.classes.DiGraph]) -> List[List[float]]:
    """
    Run and wrangle data for PageRank on each graph representing a run.
    """
    pageranks = [nx.pagerank(e) for e in graphs]
    pageranks = [list(e.values()) for e in pageranks]
    return pageranks


def enrich_experiences(experiences: List[Dict[str,
                                              Any]], scores: List[List[float]],
                       debate_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Tack scores on the final token of each experience.
    """
    for run_id, run in enumerate(experiences):
        for round_id, round in enumerate(run):
            for party_id, party in enumerate(round):
                prop_id = round_id * debate_config["num_parties"] + party
                experiences[run_id][round_id][party_id]["all_rewards"][
                    -1] = scores[run_id][prop_id]

    return experiences
