import torch
from typing import Dict, Any, List, Tuple
import networkx as nx
from accelerate import Accelerator
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, ZeroShotClassificationPipeline
import re


def reward(experiences: List[List[Dict[str, Any]]], facts: List[List[str]], debate_config: Dict[str, Any],
           nli_pipe: ZeroShotClassificationPipeline) -> Tuple[List[List[Dict[str, Any]]], List[float]]:
    """
    Coordinate the enrichment of experiences with rewards. In terms of terminology, reward in this codebase refers to the sum of KL penalties and domain-specific scores. Although the relevant aspects of debate configuration can be inferred from the structure of the `experiences` object, having to include it explicitly makes it clear that they have to share the same structure.

    Returns:
        List of enriched experiences (round x party)
        Assortative mixing values (run)
    """
    # Collapse round and party dims into a flattened props dim
    props = []
    for run_id in range(debate_config["num_debates"]):
        run_props = []
        for round_id in range(debate_config["num_rounds"]):
            for party_id in range(debate_config["num_parties"]):
                run_props += [experiences[round_id][party_id]["texts"][run_id]]

        props += [run_props]

    graphs = compose_graphs(props, facts, debate_config, nli_pipe)
    scores = compute_pagerank(graphs, debate_config)
    scores = sanitize_scores(props, scores)
    mixing = compute_mixing(graphs, debate_config)
    enriched_es = enrich_experiences(experiences, scores, debate_config)
    return enriched_es, mixing, scores, props


def compose_graphs(props: List[List[str]],
                   facts: List[List[str]],
                   debate_config: Dict[str,
                                       Any],
                   nli_pipe: ZeroShotClassificationPipeline) -> List[nx.classes.DiGraph]:
    """
    Compose a weighted directed graph using networkx where nodes represent propositions and arc represent relations of support between them.

    Returns:
        Composed graphs
    """
    assert all([len(e) == len(props[0])
                for e in props]), "Runs differ in num_props!"

    weights = compute_arc_weights(props, facts, debate_config, nli_pipe)
    graphs = []
    for run_id in range(len(props)):
        D = nx.DiGraph()
        D.add_weighted_edges_from(weights[run_id])
        graphs += [D]

    return graphs


def compute_arc_weights(
        props: List[List[str]],
        facts: List[List[str]],
        debate_config: Dict[str, Any], nli_pipe: ZeroShotClassificationPipeline) -> List[List[Tuple[int, int, float]]]:
    """
    Run pairs of props through NLI pipeline to compute arc weights for each graph. The predefined zero-shot text classification is used due to it conveniently wrapping NLI-related logic , although its original goal was to be used in a different application.

    Returns:
        List of lists of outbound-inbound-weight triples defining arcs (run x weight)
    """
    # Note: UserWarning: The sentencepiece tokenizer that you are converting
    # to a fast tokenizer uses the byte fallback option which is not
    # implemented in the fast tokenizers. In practice this means that the fast
    # version of the tokenizer can produce unknown tokens whereas the
    # sentencepiece version would have converted these unknown tokens into a
    # sequence of byte tokens matching the original piece of text.

    num_props_per_debate = debate_config["num_parties"] * \
        debate_config["num_rounds"]
    num_items_per_debate = num_props_per_debate + debate_config["num_facts"]

    weighted_edges = []
    for run_props, run_facts in zip(props, facts):
        run_items = run_props + run_facts
        run_items = [e if e != "" else "?" for e in run_items]
        run_scores = nli_pipe(
            run_items,
            run_items,
            multi_label=True,
            hypothesis_template="{}")

        run_weights = []
        for outbound_id in range(num_items_per_debate):
            for inbound_id in range(num_items_per_debate):
                if outbound_id != inbound_id and inbound_id < num_props_per_debate:
                    ref_in_id = run_scores[outbound_id]["labels"].index(run_items[inbound_id])

                    run_weights += [(outbound_id, inbound_id,
                                     round(run_scores[outbound_id]["scores"][ref_in_id], 2))]

        weighted_edges += [run_weights]

    return weighted_edges


def compute_pagerank(graphs: List[nx.classes.DiGraph],
                     debate_config: Dict[str, Any]) -> List[List[float]]:
    """
    Run and wrangle data for PageRank on each graph representing a run.

    Returns:
        List of lists of pagerank scores for propositions (run x prop)
    """
    pageranks = [nx.pagerank(e) for e in graphs]
    pageranks = [list(e.values()) for e in pageranks]

    scores = []
    for run in pageranks:
        party_avgs = []
        for party in range(debate_config["num_parties"]):
            party_sum = sum([run[party + round_id * debate_config["num_parties"]]
                            for round_id in range(debate_config["num_rounds"])])
            party_avg = party_sum / debate_config["num_rounds"]
            # Reward should be independent of prop count per party
            party_avgs += [party_avg * debate_config["num_rounds"]]

        objectives = torch.Tensor(debate_config["objectives"])
        party_avgs = torch.Tensor(party_avgs)
        adjusted_party_scores = (party_avgs @ objectives.T).tolist()

        scores += [adjusted_party_scores * debate_config["num_rounds"]]

    return scores


def sanitize_scores(props: List[List[str]], scores: List[List[float]]) -> List[List[float]]:
    for run_id, run in enumerate(props):
        for prop_id, prop in enumerate(run):
            plain = re.sub("[\.,'\!\?\-]", "", prop)
            legal = all([word.isalpha() for word in plain.split()])
            long_enough = len(plain.split()) > 4
            one_sent = len([e for e in prop if e in [".", "!", "?"]]) == 1
            if not one_sent or not legal or not long_enough:
                scores[run_id][prop_id] = 0

    return scores


def compute_mixing(graphs: List[nx.classes.DiGraph],
                   debate_config: Dict[str, Any]) -> List[float]:
    """
    Compute assortative mixing of each graph associated with a run.

    Returns:
        List of lists of assortative mixing values (run)
    """
    num_props_per_run = debate_config["num_rounds"] * debate_config[
        "num_parties"]
    mixings = []
    for G in graphs:
        # Discretize edges using mean as threshold
        G = G.copy()
        weights = [e[2] for e in G.edges.data("weight")]
        threshold = sum(weights) / len(weights)
        long_edges = list(
            filter(
                lambda e: e[2] < threshold,
                (e for e in G.edges.data('weight'))))
        le_ids = list(e[:2] for e in long_edges)
        G.remove_edges_from(le_ids)

        # Assign parties as node attributes
        for party_id in range(debate_config["num_parties"]):
            for round_id in range(debate_config["num_rounds"]):
                prop_id = party_id + round_id * debate_config["num_parties"]
                G.nodes[prop_id]["party"] = party_id
        mixings += [nx.attribute_assortativity_coefficient(G, "party")]

    return mixings


def enrich_experiences(experiences: List[List[Dict[str,
                                              Any]]], scores: List[List[float]],
                       debate_config: Dict[str, Any]) -> List[List[Dict[str,
                                                                        Any]]]:
    """
    Tack scores on the final token of each experience.

    Returns
        List of enriched experiences (round x party)
    """
    experiences = experiences.copy()
    for run_id in range(debate_config["num_debates"]):
        for round_id in range(debate_config["num_rounds"]):
            for party_id in range(debate_config["num_parties"]):
                prop_id = round_id * debate_config["num_parties"] + party_id
                experiences[round_id][party_id]["all_rewards"][run_id][-1] += scores[run_id][prop_id]
                experiences[round_id][party_id]["scores"] += [scores[run_id][prop_id]]

    return experiences
