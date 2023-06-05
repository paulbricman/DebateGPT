from inference.core import Debate
import re

def evaluate(model1, model2, num_rounds: int = 4, num_branches: int = 1):
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
    sanitized_result = []
    for br_graph in graph:
        d1_score = 0
        d2_score = 0
        sanitized_d1_score = 0
        sanitized_d2_score = 0
        valid_d1 = 0
        valid_d2 = 0
        for i in range(len(br_graph.nodes)):
            node = br_graph.nodes[i]
            # removing scores for invalid props may make sum less than one
            if(isValid(node["content"])):
                if node["party"] == 0:
                    sanitized_d1_score += node["score"]
                    valid_d1 += 1
                else:
                    sanitized_d2_score += node["score"]
                    valid_d2 += 1
            if node["party"] == 0:
                d1_score += node["score"]
            else:
                d2_score += node["score"]
        result.append([d1_score, d2_score])
        if(valid_d1 + valid_d2 > 0):
            sanitized_d1_score /= sanitized_d1_score+sanitized_d2_score
            sanitized_d2_score /= sanitized_d1_score+sanitized_d2_score
        sanitized_result.append([sanitized_d1_score, sanitized_d2_score])

    return {'raw':result, 'sanitized':sanitized_result}


def isValid(prop):
    plain = re.sub("[\\.,'\\!\\?\\-]", "", prop)
    legal = all([word.isalpha() for word in plain.split()])
    long_enough = len(plain.split()) > 4
    start_capital = long_enough and plain.strip()[0].isupper()
    one_sent = len([e for e in prop if e in [".", "!", "?"]]) == 1
    if not one_sent or not legal or not long_enough or not start_capital:
        return False
    return True

# print(evaluate("distilgpt2", "distilgpt2", num_branches=2))

def elo_rank(models, num_rounds: int = 4, num_branches: int = 1):
    elo = {}
    kfactor = 32 
    scale_factor = 400 
    exponent_base = 10
    initial_rating = 1000 
    for model in models:
        elo[model] = initial_rating
    for i in range(0, len(models)):
        for j in range(i+1, len(models)):
            model1 = models[i]
            model2 = models[j]
            res = evaluate(model1, model2, num_rounds=num_rounds, num_branches=num_branches)
            for r in res['sanitized']:
                ea = 1 / (1 + exponent_base ** ((elo[model2] - elo[model1]) / scale_factor))
                eb = 1 / (1 + exponent_base ** ((elo[model1] - elo[model2]) / scale_factor))
                if r[0] > r[1]:
                    # model1 win
                    sa = 1
                elif r[0] < r[1]:
                    # model2 win
                    sa = 0
                else:
                    #tie
                    sa = 0.5
                elo[model1] += kfactor*(sa-ea)
                elo[model2] += kfactor*(1-sa-eb)
    return elo
