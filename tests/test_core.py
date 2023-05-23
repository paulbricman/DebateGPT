from debategpt.inference.core import Debate, distance
import pytest


@pytest.fixture
def debate():
    return Debate()


def test_party_sel(debate: Debate):
    assert debate.sel_party is None
    sel = debate.party(0)
    assert sel.sel_party == 0
    sel = debate.party([0, 1])
    assert sel.sel_party == [0, 1]
    sel = debate.party(None)
    assert sel.sel_party is None


def test_round_sel(debate: Debate):
    assert debate.sel_round is None
    sel = debate.round(0)
    assert sel.sel_round == 0
    sel.curr_round = 5  # test workaround
    sel = sel.round(2, 4)
    assert sel.sel_round == (2, 4)
    sel = sel.round(None)
    assert sel.sel_round is None


def test_branch_sel(debate: Debate):
    assert debate.sel_branch is None
    sel = debate.branch(0)
    assert sel.sel_branch == 0
    debate.num_branches = 5  # test workaround
    sel = debate.branch([0, 1])
    assert sel.sel_branch == [0, 1]
    sel = debate.branch(None)
    assert sel.sel_branch is None


def test_chaining(debate: Debate):
    sel1 = debate.party(0).round(0)
    sel2 = debate.round(0).party(0)
    assert sel1.selection() == sel2.selection()

    sel1 = debate.party(1).round(0)
    sel2 = debate.round(0).party(0)
    assert sel1.selection() != sel2.selection()

    sel1 = debate.party(0).round(0).branch(0)
    sel2 = debate.branch(0).round(0).party(0)
    assert sel1.selection() == sel2.selection()

    sel1 = debate.party(0).party(0).party(0)
    sel2 = debate.party(0)
    assert sel1.selection() == sel2.selection()

    assert sel1.party(0).party(None).selection() == debate.selection()


def test_step(debate: Debate):
    assert debate.curr_party == 0
    debate.step()
    assert debate.curr_party == 1
    debate.step()
    assert debate.curr_party == 0
    assert debate.curr_round == 1
    debate.step(2)
    assert debate.curr_party == 0
    assert debate.curr_round == 2


def test_play(debate: Debate):
    d1 = debate
    d2 = debate._clone()

    d1.step(2)
    d2.play()
    assert d1.__dict__ == d2.__dict__

    d1.step(4)
    d2.play(2)
    assert d1.__dict__ == d2.__dict__


def test_fork(debate: Debate):
    debate.play(3)
    assert len(debate.prop_grid) == 1
    debate.fork()
    assert len(debate.prop_grid) == 2
    debate.fork()
    assert len(debate.prop_grid) == 4
    debate.fork(3)
    assert len(debate.prop_grid) == 12


def test_establish(debate: Debate):
    debate.play(3)
    debate.establish("This is known.")
    debate.establish(["This is also known.", "Also this is known."])
    assert len(debate.facts[0]) == 3
    debate.fork()
    assert len(debate.facts[0]) == 3
    debate.establish("Branch-invariant truth.")
    assert len(debate.facts[0]) == 4
    assert len(debate.facts[1]) == 4
    debate.establish("This fact is only established in branch 1.", 1)
    assert len(debate.facts[1]) == 5


def test_transcript(debate: Debate):
    debate.play()
    t1 = debate.transcript()
    debate.play(3)
    t2 = debate.transcript()
    debate.fork()
    t3 = debate.transcript()
    t4 = debate.round(1, 2).transcript()

    assert len(t1) > 10
    assert len(t1) < len(t2)
    assert len(t2) < len(t3)
    assert len(t4) < len(t3)


def test_render(debate: Debate):
    debate.establish(["This is established.", "This, too."])
    debate.play(3)
    debate.fork()

    r = debate.render()
    assert len(r) == debate.num_branches
    assert r[0] == r[1]
    assert len(r[0]) > 10


def test_distance(debate: Debate):
    debate.play(3)

    d1 = debate.party(0)
    d2 = debate.party(1)

    assert distance(d1, d2) == distance(d2, d1)

    prop1 = "The Earth is round."
    prop2 = "The Earth is flat."
    dist1 = distance(d1, prop1)

    assert distance(prop1, prop2) == distance(prop2, prop1)
    assert dist1 == distance(prop1, d1)
    assert dist1 != distance(d1, prop2)
    assert dist1 >= 0 and dist1 <= 1


def test_graph(debate: Debate):
    debate.establish(["The Earth is round."])
    debate.fork()
    debate.play(3)
    Gs = debate.graph()
    G = Gs[0]

    assert len(Gs) == 2
    assert G.nodes[0]["party"] == 0
    assert G.nodes[2]["round"] == 1
    assert G.nodes[1]["content"] == debate.branch(0).round(0).party(1).flattened_props()[0]
    assert G.nodes[6]["type"] == "fact"
    assert G.nodes[0]["type"] == "contribution"

    weights = [e[2] for e in G.edges(data=True)]

    assert all([e["weight"] <= 1. and e["weight"] >= 0. for e in weights])
    assert G.nodes[0]["score"] <= 1.

# test transcript loading with example transcript
def test_load(debate: Debate):
    debate.load(transcript='''A: I think we should buy gifts for everyone this Christmas. 

B: We can't afford to buy gifts for everyone! 

A: I know, but it's the thought that counts. 

B: Maybe we could make something instead? 

A: That would take too long, it's already December. 

B: We could start now and make something special. 

A: Sure, but if we have the money, why not just buy something? 

B: We don't have enough money for that. 

A: Maybe we could go in with someone else on a gift? 

B: That's a great idea! We could pool our resources and get everyone something nice.

A: Okay, let's do it!
''')
    assert debate.num_branches == 1
    assert debate.num_parties == 2
    assert debate.prop_grid[0][0][0] == 'I think we should buy gifts for everyone this Christmas.' 
    Gs = debate.graph()
    G = Gs[0]
    assert G.nodes[0]["party"] == 0
    assert G.nodes[1]["party"] == 1
    assert G.nodes[2]["round"] == 1
    