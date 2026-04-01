"""
A simple tool to execute a grid search over a hyperparameter
space

Authors
 * Artem Ploujnikov 2026
"""

import csv
import itertools
from pathlib import Path
import shlex
import subprocess
from typing import Iterable

from types import SimpleNamespace
from speechbrain.utils.logger import get_logger
from sympy import Number
from tqdm.auto import tqdm

logger = get_logger(__name__)


class GridSearch:
    def __init__(self, hparams, cmd_args=None):
        if "cwd" not in hparams:
            hparams["cwd"] = Path(".")
        if "run" not in hparams:
            hparams["run"] = "train.py"
        self.hparams = SimpleNamespace(hparams)
        self.cmd_args = cmd_args

    def __call__(self):
        """Runs the grid search"""
        self.on_search_start()
        space = enumerate_space(self.hparams.space)
        space = list(space)
        logger.info("Space size: %d experiments", len(space))
        for trial in tqdm(space):
            if self.is_finished(trial):
                logger.info(
                    "Trial %s: alrady finished, skipping",
                    format_params_log(trial)
                )
                continue
            self.run_trial(trial)

    def on_search_start(self):
        self.trials_folder = Path(self.hparams.output_folder) / "trials"
        self.trials_folder.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.params = getattr(self.hparams, "params", {})

    def get_output_folder(self, trial: dict) -> Path:
        """Finds the output folder for the specified trial

        Parameters
        ----------
        trial : dict
            The trial hyperparameters

        Returns
        -------
        output_folder : Path
            The output path
        """
        suffix = format_params_path(trial)
        trial_name = f"trial--{suffix}"
        output_folder = self.trials_folder / trial_name
        output_folder.mkdir(parents=True, exist_ok=True)
        return output_folder

    def run_trial(self, trial: dict):
        """Runs a single grid search trial

        Parameters
        ----------
        trial : dict
            Trial hyperparameters
        """
        output_folder = self.get_output_folder(trial)
        logger.info("Grid Search: Running trial: %s", format_params_log(trial))
        params = {
            **self.params,
            "output_folder": str(output_folder),
            **trial
        }
        cmd_args = format_params_cmd(params)
        run = self.hparams.run
        if isinstance(run, str):
            run = shlex.split(run)
        cmd = (
            run
            + self.cmd_args
            + cmd_args
        )
        cmd_str = shlex.join(cmd)
        logger.info("Running %s", cmd_str)
        result = subprocess.run(cmd)
        if result.returncode == 0:
            self.write_metrics(trial)
            self.mark_finished(trial)
        else:
            logger.warning(
                "Command could not complete and returned exit code %d",
                result.returncode
            )

    def get_metrics(self, trial: dict) -> dict:
        """Retrieves the metric values for the specified trial

        Arguments
        ---------
        trial : dict
            The trial hyperparameters

        Returns
        -------
        stats : dict
            The statistics dictionary
        """
        output_folder = self.get_output_folder(trial)
        train_log_file_name = output_folder / "train_log.txt"
        if not train_log_file_name.exists():
            logger.warning("%s does not exist", train_log_file_name)
            return {key: None for key in self.hparams.metrics}
        line = None
        with open(train_log_file_name) as train_log:
            for line in train_log:
                pass
        if line is None:
            logger.warning("No metrics found in %s ", train_log_file_name)
            return {key: None for key in self.hparams.metrics}
        log_metrics = parse_train_log_data(line)
        metrics = {key: log_metrics.get(f"{self.hparams.stage}_{key}") for key in self.hparams.metrics}
        return metrics

    def write_metrics(self, trial: dict):
        """Ouputs the statistics for the specified trial

        Arguments
        ---------
        trial : dict
            The trial hyperparameters
        """
        header = list(self.hparams.space.keys()) + self.hparams.metrics
        metrics_file_name = Path(self.hparams.output_folder) / "metrics.csv"
        is_fresh = not metrics_file_name.exists()
        # TODO: Fetch metrics
        metrics = self.get_metrics(trial)
        with open(metrics_file_name, "a") as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=header)
            if is_fresh:
                writer.writeheader()
            row = {**trial, **metrics}
            writer.writerow(row)

    def mark_finished(self, trial: dict):
        """Marks a trial as finished

        Arguments
        ---------
        trial : dict
            Trial hyperparameters
        """
        file_name = self.get_finished_file_name(trial)
        file_name.touch()

    def is_finished(self, trial: dict) -> bool:
        """Determines whether the specified trial is
        finished

        Parameters
        ----------
        trial : dict
            Trial hyperparameters

        Returns
        -------
        is_finished : bool
            Whether the trial is finished
        """
        file_name = self.get_finished_file_name(trial)
        is_finished = file_name.exists()
        return is_finished

    def get_finished_file_name(self, trial: dict) -> Path:
        """Gets the file name for the marker indicating whether
        a trial is finished
        
        Returns
        -------
        file_name : Path
            the file name, with the path
        """
        output_folder = self.get_output_folder(trial)
        return output_folder / ".finished"


def parse_train_log_data(data: str) -> dict:
    """Parses a line from the training log into a dictionary

    Arguments
    ---------
    data : str
        The log line

    Returns
    -------
    result : dict
        The metrics from the log line
    """
    parts = [
        part
        for chunk in data.split(" - ")
        for part in chunk.split(", ")
    ]
    raw_data = dict([part.split(": ") for part in parts])
    return {
        key.replace(" ", "_"): convert_numeric(value)
        for key, value in raw_data.items()
    }


def convert_numeric(value: str) -> str | Number:
    """If the value is numeric, it will be converted to
    an int or a float. Otherwise it will remain a string

    Arguments
    ---------
    value : str
        A string value

    Returns
    -------
    value : str | Number
        A"""
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            pass
    return value


def format_params_log(params: dict) -> str:
    """Formats params for display in logs

    Arguments
    ---------
    params : dict
        A dictionary with parameters

    Returns
    -------
    result: str
        The formatted parameters
    """
    return ", ".join(
        f"{key} = {value}"
        for key, value in params.items()
    )


def format_params_path(params: dict) -> str:
    """Formats parameters for use in paths

    Arguments
    ---------
    params : dict
        A dictionary with parameters

    Returns
    -------
    result : str
        A formatted path
    """
    return "--".join(
        f"{key}-{value}"
        for key, value in params.items()
    )


def format_params_cmd(params: dict) -> list[str]:
    """Converts a dictionary of parameters to a list of
    command-line arguments

    Arguments
    ---------
    params : dict
        A dictionary of parameters

    Returns
    -------
    result : list[str]
        a list of arguments (e.g. to be appended)
        to a command in `subprocess.run`"""
    args = []
    for key, value in params.items():
        args.append(f"--{key}")
        args.append(str(value))
    return args


def enumerate_space(space: dict) -> Iterable:
    """Enumerates the search space

    Arguments
    ---------
    space : dict
        The search space

    Returns
    -------
    result : Iterable
        An enumeration of experiments to run
    """
    keys = list(space.keys())
    values = itertools.product(
        *space.values()
    )
    return (
        dict(zip(keys, value))
        for value in values
    )
