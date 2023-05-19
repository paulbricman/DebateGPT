from inference.core import Debate

def evaluate(model1, model2, num_rounds: int = 4, num_branches: int = 2):
    """
    Evaluates the debate capabilities of two models against each other
    """
    d1 = Debate(model=model1)
    d2 = Debate(model=model2)
    d1.fork(num_branches)
    d2.fork(num_branches)
    for round in range(num_rounds):
        d1.curr_party = 0
        d1.curr_round = round
        d2.curr_round = round
        d1.step()
        # print(d2.prop_grid)
        for branch in range(num_branches):
            d2.prop_grid[branch][round].append(d1.prop_grid[branch][round][0])
        d2.curr_party = 1
        d2.step()
        for branch in range(num_branches):
            d1.prop_grid[branch][round].append(d2.prop_grid[branch][round][1])
            if round != num_rounds-1:
                d1.prop_grid[branch].append([])
    graph = d1.graph()
    result = []
    for br_graph in graph:
        d1_score = 0
        d2_score = 0
        for i in range(len(br_graph.nodes)):
            node = br_graph.nodes[i]
            if node["party"] == 0:
                d1_score += node["score"]
            else:
                d2_score += node["score"]
        result.append([d1_score, d2_score])

    return result

print(evaluate("distilgpt2","distilgpt2"))

