from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "experiments" / "run_hyperparam_sweep_hpc.py"
DATASET_PATH = ROOT / "htru2" / "HTRU_2.arff"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class RunHyperparamSweepHpcTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module("run_hyperparam_sweep_hpc", SCRIPT_PATH)

    def test_build_sweep_specs_is_deterministic_and_matches_cartesian_product_size(self):
        specs = self.module.build_sweep_specs()
        expected_count = (
            len(self.module.ARCHITECTURE_CANDIDATES)
            * len(self.module.TRAIN_FRACTION_OPTIONS)
            * len(self.module.LEARNING_RATE_OPTIONS)
            * len(self.module.POSITIVE_CLASS_WEIGHT_OPTIONS)
            * len(self.module.INIT_SEED_OPTIONS)
        )

        self.assertEqual(len(specs), expected_count)
        self.assertEqual(
            specs[0],
            self.module.SweepSpec(
                architecture_name=self.module.ARCHITECTURE_CANDIDATES[0][0],
                architecture_shape=self.module.ARCHITECTURE_CANDIDATES[0][1],
                train_fraction=self.module.TRAIN_FRACTION_OPTIONS[0],
                learning_rate=self.module.LEARNING_RATE_OPTIONS[0],
                positive_class_weight=self.module.POSITIVE_CLASS_WEIGHT_OPTIONS[0],
                init_seed=self.module.INIT_SEED_OPTIONS[0],
            ),
        )
        self.assertEqual(
            specs[-1],
            self.module.SweepSpec(
                architecture_name=self.module.ARCHITECTURE_CANDIDATES[-1][0],
                architecture_shape=self.module.ARCHITECTURE_CANDIDATES[-1][1],
                train_fraction=self.module.TRAIN_FRACTION_OPTIONS[-1],
                learning_rate=self.module.LEARNING_RATE_OPTIONS[-1],
                positive_class_weight=self.module.POSITIVE_CLASS_WEIGHT_OPTIONS[-1],
                init_seed=self.module.INIT_SEED_OPTIONS[-1],
            ),
        )

    def test_available_core_count_prefers_sched_getaffinity(self):
        with patch.object(self.module.os, "sched_getaffinity", return_value={0, 1, 2}, create=True):
            with patch.object(self.module.os, "cpu_count", return_value=99):
                self.assertEqual(self.module.available_core_count(), 3)

    def test_available_core_count_falls_back_to_cpu_count(self):
        with patch.object(self.module.os, "sched_getaffinity", side_effect=OSError("blocked"), create=True):
            with patch.object(self.module.os, "cpu_count", return_value=7):
                self.assertEqual(self.module.available_core_count(), 7)

    def test_round_robin_partition_covers_each_item_once(self):
        items = list(range(13))
        partitions = [
            self.module.partition_round_robin(items, partition_index=index, partition_count=4)
            for index in range(4)
        ]
        flattened = [item for partition in partitions for item in partition]

        self.assertCountEqual(flattened, items)
        self.assertEqual(len(flattened), len(items))
        self.assertEqual(sum(item in partitions[0] for item in items), 4)

    def test_spec_directory_name_is_stable_and_sanitized(self):
        spec = self.module.SweepSpec(
            architecture_name="very large tapered",
            architecture_shape=(256, 128, 64, 8, 1),
            train_fraction=0.80,
            learning_rate=0.03,
            positive_class_weight=2.0,
            init_seed=23,
        )

        self.assertEqual(
            self.module.spec_directory_name(spec),
            "arch-very_large_tapered__split-0p80__lr-0p03__pcw-2p0__seed-23",
        )

    def test_smoke_run_writes_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "experiment_hpc_smoke"
            with patch.object(
                self.module,
                "ARCHITECTURE_CANDIDATES",
                (("tiny_shallow", (4, 1)),),
            ), patch.object(
                self.module,
                "TRAIN_FRACTION_OPTIONS",
                (0.70,),
            ), patch.object(
                self.module,
                "LEARNING_RATE_OPTIONS",
                (0.01,),
            ), patch.object(
                self.module,
                "POSITIVE_CLASS_WEIGHT_OPTIONS",
                (1.0,),
            ), patch.object(
                self.module,
                "INIT_SEED_OPTIONS",
                (11,),
            ), patch.object(
                self.module,
                "DEFAULT_MILESTONES",
                (1, 2),
            ):
                specs = self.module.build_sweep_specs()
                stats_paths = self.module.run_local_sweep(
                    output_dir=output_dir,
                    dataset_path=DATASET_PATH,
                    jobs=1,
                    specs=specs,
                )

            self.assertEqual(len(stats_paths), 1)
            artifact_dir = self.module.artifact_dir_for_spec(output_dir, specs[0])
            self.assertTrue((artifact_dir / "hyperparams.json").exists())
            self.assertTrue((artifact_dir / "dataset_split.json").exists())
            self.assertTrue((artifact_dir / "training_history.json").exists())
            self.assertTrue((artifact_dir / "model.ffnnpy").exists())
            self.assertTrue((artifact_dir / "model_stats.json").exists())


if __name__ == "__main__":
    unittest.main()
