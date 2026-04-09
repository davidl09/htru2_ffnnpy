from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
STATS_RUNNER_PATH = ROOT / "experiments" / "run_saved_model_stats_hpc.py"
SWEEP_SCRIPT_PATH = ROOT / "experiments" / "run_hyperparam_sweep_hpc.py"
DATASET_PATH = ROOT / "htru2" / "HTRU_2.arff"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class RunSavedModelStatsHpcTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module("run_saved_model_stats_hpc", STATS_RUNNER_PATH)
        cls.sweep_module = load_module("run_hyperparam_sweep_hpc_for_stats_helper", SWEEP_SCRIPT_PATH)

    def test_discover_model_paths_finds_and_sorts_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "sweep"
            first_dir = output_dir / "b_model"
            second_dir = output_dir / "a_model"
            first_dir.mkdir(parents=True)
            second_dir.mkdir(parents=True)
            (first_dir / "model.ffnnpy").write_text("x", encoding="utf-8")
            (second_dir / "model.ffnnpy").write_text("y", encoding="utf-8")

            model_paths = self.module.discover_model_paths(output_dir)

            self.assertEqual(
                model_paths,
                [
                    second_dir / "model.ffnnpy",
                    first_dir / "model.ffnnpy",
                ],
            )

    def test_smoke_run_regenerates_stats_from_sweep_output_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "experiment_hpc_smoke"
            with patch.object(
                self.sweep_module,
                "ARCHITECTURE_CANDIDATES",
                (("tiny_shallow", (4, 1)),),
            ), patch.object(
                self.sweep_module,
                "TRAIN_FRACTION_OPTIONS",
                (0.70,),
            ), patch.object(
                self.sweep_module,
                "LEARNING_RATE_OPTIONS",
                (0.01,),
            ), patch.object(
                self.sweep_module,
                "POSITIVE_CLASS_WEIGHT_OPTIONS",
                (1.0,),
            ), patch.object(
                self.sweep_module,
                "INIT_SEED_OPTIONS",
                (11,),
            ), patch.object(
                self.sweep_module,
                "DEFAULT_MILESTONES",
                (1, 2),
            ):
                specs = self.sweep_module.build_sweep_specs()
                self.sweep_module.run_local_sweep(
                    output_dir=output_dir,
                    dataset_path=DATASET_PATH,
                    jobs=1,
                    specs=specs,
                )

            model_path = self.sweep_module.model_path_for_spec(output_dir, specs[0])
            stats_path = model_path.with_name("model_stats.json")
            stats_path.unlink()

            written_stats_paths = self.module.run_stats_over_output_dir(
                output_dir=output_dir,
                dataset_path=DATASET_PATH,
                jobs=1,
            )

            self.assertEqual(written_stats_paths, [stats_path])
            self.assertTrue(stats_path.exists())

    def test_bundle_download_writes_viewer_and_zip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "experiment_hpc_smoke"
            with patch.object(
                self.sweep_module,
                "ARCHITECTURE_CANDIDATES",
                (("tiny_shallow", (4, 1)),),
            ), patch.object(
                self.sweep_module,
                "TRAIN_FRACTION_OPTIONS",
                (0.70,),
            ), patch.object(
                self.sweep_module,
                "LEARNING_RATE_OPTIONS",
                (0.01,),
            ), patch.object(
                self.sweep_module,
                "POSITIVE_CLASS_WEIGHT_OPTIONS",
                (1.0,),
            ), patch.object(
                self.sweep_module,
                "INIT_SEED_OPTIONS",
                (11,),
            ), patch.object(
                self.sweep_module,
                "DEFAULT_MILESTONES",
                (1, 2),
            ):
                specs = self.sweep_module.build_sweep_specs()
                self.sweep_module.run_local_sweep(
                    output_dir=output_dir,
                    dataset_path=DATASET_PATH,
                    jobs=1,
                    specs=specs,
                )

            written_stats_paths = self.module.run_stats_over_output_dir(
                output_dir=output_dir,
                dataset_path=DATASET_PATH,
                jobs=1,
                bundle_download=True,
            )

            self.assertEqual(len(written_stats_paths), 1)
            viewer_path = self.module.default_bundle_viewer_path(output_dir)
            zip_path = self.module.default_bundle_zip_path(output_dir)
            self.assertTrue(viewer_path.exists())
            self.assertTrue(zip_path.exists())

            with ZipFile(zip_path) as archive:
                self.assertEqual(
                    archive.namelist(),
                    [
                        self.module.DEFAULT_VIEWER_FILENAME,
                        str(written_stats_paths[0].relative_to(output_dir)),
                    ],
                )


if __name__ == "__main__":
    unittest.main()
