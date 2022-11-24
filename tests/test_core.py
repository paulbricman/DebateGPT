from debategpt.inference.core import Debate, distance
import pytest


@pytest.fixture
def debate():
    return Debate()


def test_party_sel(debate: Debate):
    assert debate.sel_party == None
    sel = debate.party(0)
    assert sel.sel_party == 0
    sel = debate.party([0, 1])
    assert sel.sel_party == [0, 1]
    sel = debate.party(None)
    assert sel.sel_party == None


def test_round_sel(debate: Debate):
    assert debate.sel_round == None
    sel = debate.round(0)
    assert sel.sel_round == 0
    sel.curr_round = 5 # test workaround
    sel = sel.round(2, 4)
    assert sel.sel_round == (2, 4)
    sel = sel.round(None)
    assert sel.sel_round == None


def test_branch_sel(debate: Debate):
    assert debate.sel_branch == None
    sel = debate.branch(0)
    assert sel.sel_branch == 0
    sel.num_branches = 5 # test workaround
    sel = debate.branch([0, 1])
    assert sel.sel_branch == [0, 1]
    sel = debate.branch(None)
    assert sel.sel_branch == None


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

