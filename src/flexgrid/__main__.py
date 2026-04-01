"""A simple tool to execute a grid search over a hyperparameter
space

Script entry point

Authors
 * Artem Ploujnikov 2026
"""

import speechbrain as sb
from flexgrid.utils import split_args
from flexgrid.search import GridSearch
from hyperpyyaml import load_hyperpyyaml

if __name__ == "__main__":
    own_args, cmd_args = split_args()
    hparams_file, run_opts, overrides = sb.parse_arguments(own_args)
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f, overrides=overrides)
    sb.create_experiment_directory(
        hparams["output_folder"],
        hparams_file,
        overrides=overrides,
    )

    grid_search = GridSearch(hparams, cmd_args)
    grid_search()
