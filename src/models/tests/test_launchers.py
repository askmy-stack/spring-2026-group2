from __future__ import annotations

from argparse import Namespace

from models.baseline import _launcher as baseline_launcher
from models.improved import _launcher as improved_launcher


def test_baseline_launcher_routes_to_train(monkeypatch):
    captured: dict[str, object] = {}

    def fake_build_parser(_description: str):
        class _Parser:
            def parse_args(self, argv):
                captured["argv"] = argv
                return Namespace(model=argv[1], sentinel="train")

        return _Parser()

    def fake_train_model(model_name: str, args: Namespace):
        captured["model_name"] = model_name
        captured["args"] = args
        return 17

    monkeypatch.setattr(baseline_launcher, "build_train_parser", fake_build_parser)
    monkeypatch.setattr(baseline_launcher, "train_model", fake_train_model)

    code = baseline_launcher.run_model_command("deepconvnet", "train", ["--epochs", "3"])
    assert code == 17
    assert captured["model_name"] == "deepconvnet"
    assert captured["argv"] == ["--model", "deepconvnet", "--epochs", "3"]


def test_improved_launcher_routes_to_eval(monkeypatch):
    captured: dict[str, object] = {}

    def fake_build_parser(_description: str):
        class _Parser:
            def parse_args(self, argv):
                captured["argv"] = argv
                return Namespace(model=argv[1], sentinel="eval")

        return _Parser()

    def fake_eval_model(model_name: str, args: Namespace):
        captured["model_name"] = model_name
        captured["args"] = args
        return 23

    monkeypatch.setattr(improved_launcher, "build_eval_parser", fake_build_parser)
    monkeypatch.setattr(improved_launcher, "eval_model", fake_eval_model)

    code = improved_launcher.run_model_command("st_eegformer", "eval", ["--split", "test"])
    assert code == 23
    assert captured["model_name"] == "st_eegformer"
    assert captured["argv"] == ["--model", "st_eegformer", "--split", "test"]


def test_launcher_rejects_invalid_mode():
    try:
        improved_launcher.run_model_command("st_eegformer", "invalid")
    except ValueError as exc:
        assert "Unsupported mode" in str(exc)
    else:
        raise AssertionError("Expected invalid launcher mode to raise ValueError.")
