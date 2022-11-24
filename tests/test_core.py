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
    sel.num_branches = 5  # test workaround
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


def test_distance(debate: Debate):
    distance(debate, debate)
    distance("Test string", debate)
    distance(debate, "Test string")


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
