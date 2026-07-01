"""
promote_if_better() ve check_latest_accuracy() icin testler.
Gercek MLflow registry'ye dokunmamak icin MlflowClient mock'lanir.
"""
from unittest.mock import MagicMock, patch
import json
import auto_retrain


def make_version(version, stage, run_id):
    v = MagicMock()
    v.version = str(version)
    v.current_stage = stage
    v.run_id = run_id
    return v


def make_run(accuracy):
    run = MagicMock()
    run.data.metrics = {"accuracy": accuracy}
    return run


@patch("auto_retrain.MlflowClient")
def test_no_versions_in_registry_skips_promotion(mock_client_cls):
    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = []
    mock_client_cls.return_value = mock_client

    auto_retrain.promote_if_better()

    mock_client.transition_model_version_stage.assert_not_called()


@patch("auto_retrain.MlflowClient")
def test_first_model_promoted_when_no_production(mock_client_cls):
    mock_client = MagicMock()
    v1 = make_version(1, "None", "run1")
    mock_client.search_model_versions.return_value = [v1]
    mock_client.get_run.return_value = make_run(0.80)
    mock_client_cls.return_value = mock_client

    auto_retrain.promote_if_better()

    mock_client.transition_model_version_stage.assert_called_once_with(
        name=auto_retrain.MODEL_NAME, version="1", stage="Production"
    )


@patch("auto_retrain.MlflowClient")
def test_challenger_better_promotes_and_archives_champion(mock_client_cls):
    mock_client = MagicMock()
    champion = make_version(2, "Production", "run_champ")
    challenger = make_version(3, "None", "run_chal")
    mock_client.search_model_versions.return_value = [champion, challenger]

    def get_run_side_effect(run_id):
        if run_id == "run_champ":
            return make_run(0.80)
        return make_run(0.85)

    mock_client.get_run.side_effect = get_run_side_effect
    mock_client_cls.return_value = mock_client

    auto_retrain.promote_if_better()

    calls = mock_client.transition_model_version_stage.call_args_list
    assert len(calls) == 2
    mock_client.transition_model_version_stage.assert_any_call(
        name=auto_retrain.MODEL_NAME, version="3", stage="Production"
    )
    mock_client.transition_model_version_stage.assert_any_call(
        name=auto_retrain.MODEL_NAME, version="2", stage="Archived"
    )


@patch("auto_retrain.MlflowClient")
def test_challenger_worse_blocks_promotion(mock_client_cls):
    mock_client = MagicMock()
    champion = make_version(2, "Production", "run_champ")
    challenger = make_version(3, "None", "run_chal")
    mock_client.search_model_versions.return_value = [champion, challenger]

    def get_run_side_effect(run_id):
        if run_id == "run_champ":
            return make_run(0.8022)
        return make_run(0.7973)

    mock_client.get_run.side_effect = get_run_side_effect
    mock_client_cls.return_value = mock_client

    auto_retrain.promote_if_better()

    mock_client.transition_model_version_stage.assert_not_called()


def test_check_latest_accuracy_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert auto_retrain.check_latest_accuracy() == 0.0


def test_check_latest_accuracy_valid_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "last_accuracy.json").write_text(
        json.dumps({"accuracy": 0.9123}), encoding="utf-8"
    )
    assert auto_retrain.check_latest_accuracy() == 0.9123


def test_check_latest_accuracy_corrupted_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "last_accuracy.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    assert auto_retrain.check_latest_accuracy() == 0.0
