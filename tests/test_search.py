import csv
import io
from pathlib import Path
import tempfile
import unittest

from flexgrid.search import GridSearch, parse_train_log_data


class ParseTrainLogDataTest(unittest.TestCase):
    def test_list_metric_remains_a_string(self):
        precisions = (
            "[50.327011118378024, 27.819916809395643, "
            "15.900222965440356, 9.41024814695456]"
        )
        metrics = parse_train_log_data(
            f"epoch: 1 - test bleu: 19.5, test bleu_precisions: {precisions}"
        )

        self.assertEqual(metrics["epoch"], 1)
        self.assertEqual(metrics["test_bleu"], 19.5)
        self.assertEqual(metrics["test_bleu_precisions"], precisions)
        self.assertIsInstance(metrics["test_bleu_precisions"], str)

    def test_list_metric_can_be_written_to_csv(self):
        value = "[50.0, 27.0, 15.0, 9.0]"
        metrics = parse_train_log_data(f"test bleu_precisions: {value}")
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["test_bleu_precisions"])
        writer.writeheader()
        writer.writerow(metrics)

        output.seek(0)
        row = next(csv.DictReader(output))
        self.assertEqual(row["test_bleu_precisions"], value)


class IsFinishedTest(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.search = GridSearch.__new__(GridSearch)
        self.search.trials_folder = Path(self.temp_directory.name)
        self.trial = {"learning_rate": 0.1}

    def test_epoch_loaded_line_is_a_finished_fallback(self):
        output_folder = self.search.get_output_folder(self.trial)
        (output_folder / "train_log.txt").write_text(
            "epoch: 1, train loss: 2.0\n"
            "ePoCh LoAdEd: 10\n"
        )

        self.assertTrue(self.search.is_finished(self.trial))

    def test_text_later_in_line_does_not_mark_trial_finished(self):
        output_folder = self.search.get_output_folder(self.trial)
        (output_folder / "train_log.txt").write_text(
            "status: Epoch loaded: 10\n"
        )

        self.assertFalse(self.search.is_finished(self.trial))


if __name__ == "__main__":
    unittest.main()
