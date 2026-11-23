"""Named configurations for screening, and the validation mazes used to choose between them.

A configuration = training set (multimaze/training_sets.py) + how the mazes are combined
(--combine) + gene ranges (variants/variant_defs.py). Add a line to CONFIGS to try another one.

The validation mazes (valid_* in gridmaze.LAYOUTS) are never used for training and are ONLY used
to choose between configurations. The unseen_* mazes and the scaled copies stay untouched until
the final comparison, so the chosen configuration is not tested on the mazes it was chosen with.
"""

VALIDATION_MAZES = ["valid_7x6", "valid_9x8", "valid_10x7", "valid_11x10", "valid_13x9"]

CONFIGS = {
    # reference: the original experiment (one maze, normal ranges)
    "single":        dict(set="single", combine="mean", variant="default"),
    # multi-maze (step 2)
    "A":             dict(set="A", combine="mean", variant="default"),
    "A2":            dict(set="A2", combine="mean", variant="default"),      # 2 runs per maze
    "A_relative":    dict(set="A", combine="relative", variant="default"),   # every maze counts the same
    "A_worst":       dict(set="A", combine="worst", variant="default"),      # the slowest maze counts
    # gene ranges (variants/)
    "single_wide":   dict(set="single", combine="mean", variant="wide_progress"),
    "A_wide":        dict(set="A", combine="mean", variant="wide_progress"),
    "single_noprog": dict(set="single", combine="mean", variant="no_progress"),
    "A_noprog":      dict(set="A", combine="mean", variant="no_progress"),
    # later, on the server: the 2x maze in the training set
    "B":             dict(set="B", combine="mean", variant="default"),
}

# screened when no --configs is given; the first one is the reference for the tests
DEFAULT = ["single", "A", "A2", "A_relative", "A_worst", "A_wide"]

# relative cost of one fitness evaluation (single-maze = 1.0), measured or estimated
COST = {"single": 1.0, "A": 1.2, "A2": 2.4, "B": 2.5}
