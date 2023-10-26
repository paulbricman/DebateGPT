# DebateGPT

Implementation of the initial [_ArgRank_](https://compphil.github.io/truth/#argrank) and [_DebateGPT_](https://compphil.github.io/truth/#obtaining-debategpt) prototypes, used in [the experiments](https://compphil.github.io/truth/#benchmarking-argranks-dependencies) conducted during [AI Safety Camp 8](https://aisafety.camp/).

This repo has been made public almost a year from [the initial commit](https://github.com/paulbricman/DebateGPT/commit/b2728f2b06b5ac69dd5527658b8e502838f79e41), time during which its attentional hazardousness has decreased with the _ChatGPT_ craze. For up-to-date information on follow-up work, please refer to the homepage of [the broader research agenda](https://paulbricman.com/defensibility/).

## Inference primitives

```python
from debategpt.inference.core import Debate, distance

d = Debate()

# Advance the debate two full rounds (i.e. each party has one contribution).
d.play(2)

# Advance the debate two steps (i.e. two contributions in total). With two parties, this is equivalent to a full round.
d.step(2)

# Render human-readable debate transcript.
print(d.transcript())

# Introduces propositions in the debate which are not "owned" by any party. These can be seen as observations about the world.
d.establish("The Earth is round.")

# The following splits the debate into parallel branches. Forking can be repeated and interweaved with establishing facts, advancing the debate, etc.
d.fork(4)

# Make specific selections of parts of the debate(s).
sel1 = d.branch(1).party(0).round(0, 2)
sel2 = d.party([0, 1]).round(1)
sel3 = d.round(2).branch([0, 1]).party(0)

# First selector narrows in on two utterances.
assert len(sel1.flattened_props()) == 1 * 1 * 2

# When selector is not specified (e.g. branch here), all elements are considered.
assert len(sel2.flattened_props()) == 4 * 1 * 2

# Selector order doesn't really matter.
assert len(sel3.flattened_props()) == 2 * 1 * 1

# Selectors can then be plugged into the distance function, which averages distances between (ordered) pairs of propositions.
dist1 = distance(sel1, sel2)
dist2 = distance(sel2, sel3)
dist3 = distance(sel1, sel3)

# Extract the argument graph associated with a selector. This can then be used with tools from the `networkx` package.
G = sel1.graph
```

## Training folder structure

- `debategpt.training.orchestrator` manages high-level transcript generation, populates the experience store, and handles weight updates.
- `debategpt.training.reward` implements [_ArgRank_](https://compphil.github.io/truth/#argrank) and helps evaluate generated transcripts.
- `debategpt.training.trainer` gets the two elements above to work in concert.
- `scripts/train.py` minimally wraps the trainer above with default settings.
