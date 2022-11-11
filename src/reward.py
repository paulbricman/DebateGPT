from typing import Dict, Any, List
import networkx as nx
from accelerate import Accelerator
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def reward(experiences: List[List[Dict[str, Any]]],
           debate_config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    Coordinate the enrichment of experiences with rewards. In terms of terminology, reward in this codebase refers to the sum of KL penalties and domain-specific scores. Although the relevant aspects of debate configuration can be inferred from the structure of the `experiences` object, having to include it explicitly makes it clear that they have to share the same structure.
    """
    # Collapse round and party dims into a flattened props dim
    props = []
    for run_id in range(debate_config["num_debates"]):
        run_props = []
        for round_id in range(debate_config["num_rounds"]):
            for party_id in range(debate_config["num_parties"]):
                run_props += [experiences[round_id][party_id]["texts"][run_id]]

        props += [run_props]

    graphs = compose_graphs(props)
    scores = compute_pagerank(graphs)
    mixing = compute_mixing(graphs)
    enriched_es = enrich_experiences(experiences, scores)
    return enriched_es, mixing


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


def compute_pagerank(graphs: List[nx.classes.DiGraph], debate_config: Dict[str, Any]) -> List[List[float]]:
    """
    Run and wrangle data for PageRank on each graph representing a run.
    """
    pageranks = [nx.pagerank(e) for e in graphs]
    pageranks = [list(e.values()) for e in pageranks]

    scores = []
    for run in pageranks:
        party_avgs = []
        for party in range(debate_config["num_parties"]):
            party_sum = sum([run[party + round_id * debate_config["num_parties"] for round_id in range(debate_config["num_rounds"])])
            party_avg = party_sum / debate_config["num_rounds"]
            party_avgs += [party_avg]

        objectives = torch.Tensor(debate_config["objectives"])
        party_avgs = torch.Tensor(party_avgs)
        adjusted_party_scores = ((party_avgs @ objectives).tolist))

        scores += [adjusted_party_scores * debate_config["num_rounds"]]

    return scores


def compute_mixing(graphs: List[nx.classes.DiGraph], debate_config: Dict[str, Any]) -> List[List[float]]:
    num_props_per_run = debate_config["num_rounds"] * debate_config[
        "num_parties"]
    mixings = []
    for G in graphs:
        # Assign parties as node attributes
        for party_id in range(debate_config["num_parties"]):
            for round_id in range(debate_config["num_rounds"]):
                prop_id = party_id + round_id * debate_config["num_parties"]
                G.nodes[prop_id]["party"] = party
        mixings += [nx.attribute_assortativity_coefficient(G, "party")]

    return mixings

def enrich_experiences(experiences: List[Dict[str,
                                              Any]], scores: List[List[float]],
                       debate_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Tack scores on the final token of each experience.
    """
    for run_id in range(debate_config["num_debates"]):
        for round_id in range(debate_config["num_rounds"]):
            for party_id in range(debate_config["num_parties"]):
                prop_id = round_id * debate_config["num_parties"] + party
                experiences[round_id][party_id]["all_rewards"][run_id][-1] += scores[run_id][prop_id]
                experiences[round_id][party_id]["scores"][run_id][-1] = scores[run_id][prop_id]

    return experiences
