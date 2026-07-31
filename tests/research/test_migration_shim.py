"""Focused tests for staged root-script and legacy artifact migration."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import warnings

import pytest
import torch

from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.migration import (
    EVALUATE_CLI_DEFAULTS,
    TRAIN_CLI_DEFAULTS,
    assert_research_import_boundary,
    convert_legacy_checkpoint,
    convert_legacy_cli_arguments,
    dispatch_legacy_script,
    find_forbidden_research_imports,
    load_legacy_checkpoint,
    resolve_legacy_symbol,
    resolve_script_shim,
)
from pursuit_evasion_rl.research.policies.baselines import GoalEvader
from pursuit_evasion_rl.research.variants.observations import Observation28DAdapter

pytestmark = pytest.mark.offline
ROOT = Path(__file__).resolve().parents[2]
GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures/migration_golden.json").read_text(encoding="utf-8")
)


def _checkpoint_payload():
    return {
        "police_actor": {
            "network.0.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)
        },
        "police_critic": {"network.0.bias": torch.tensor([1.5], dtype=torch.float32)},
        "police_optimizer": {
            "state": {},
            "param_groups": [{"lr": 0.0003, "params": [0]}],
        },
        "epoch": 12,
        "config": {"observation": "osm_topology_v1", "num_police": 6},
        "best_capture_rate": 0.99,
        "notes": None,
    }


def test_legacy_cli_arguments_are_lossless_and_match_golden_hashes():
    train_args = dict(TRAIN_CLI_DEFAULTS)
    train_args.update(GOLDEN["train"]["overrides"])
    evaluation_args = dict(EVALUATE_CLI_DEFAULTS)
    evaluation_args.update(GOLDEN["evaluate"]["overrides"])

    train = convert_legacy_cli_arguments("train_osm_pursuit.py", train_args)
    evaluation = convert_legacy_cli_arguments("eval_trained", evaluation_args)

    assert dict(train.arguments) == train_args
    assert dict(train.canonical_config["legacy_arguments"]) == train_args
    assert dict(train.canonical_config["extra_arguments"]) == {"custom_tag": "golden"}
    assert train.content_hash == GOLDEN["train"]["conversion_hash"]
    assert evaluation.content_hash == GOLDEN["evaluate"]["conversion_hash"]
    assert dict(evaluation.arguments) == evaluation_args
    assert not train.research_trainer_eligible
    assert not evaluation.research_trainer_eligible


def test_checkpoint_metadata_and_state_identity_match_golden_fixture(tmp_path):
    payload = _checkpoint_payload()
    source_hash = GOLDEN["checkpoint"]["source_content_hash"]

    converted = convert_legacy_checkpoint(payload, source_content_hash=source_hash)
    reordered = convert_legacy_checkpoint(
        dict(reversed(tuple(payload.items()))), source_content_hash=source_hash
    )

    assert converted.content_hash == GOLDEN["checkpoint"]["conversion_hash"]
    assert reordered.content_hash == converted.content_hash
    assert converted.metadata == {
        "epoch": 12,
        "config": {"observation": "osm_topology_v1", "num_police": 6},
        "best_capture_rate": 0.99,
        "notes": None,
    }
    actor = converted.state_fingerprints["police_actor"]["network.0.weight"]
    assert actor["kind"] == "tensor"
    assert actor["dtype"] == "torch.float32"
    assert actor["shape"] == (2, 3)
    assert len(actor["sha256"]) == 64
    assert not converted.complete_resume_state

    checkpoint = tmp_path / "legacy.pt"
    torch.save(payload, checkpoint)
    loaded = load_legacy_checkpoint(checkpoint)
    assert loaded.source_content_hash == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert loaded.metadata == converted.metadata
    assert loaded.state_fingerprints == converted.state_fingerprints


def test_import_compatibility_resolves_package_owners_without_root_imports():
    assert resolve_legacy_symbol("demo_pursuit", "GoalEvader") is GoalEvader
    assert resolve_legacy_symbol("osm_obs_aug", "AugmentedObs") is Observation28DAdapter
    with pytest.raises(ResearchValidationError, match="symbol is not exported"):
        resolve_legacy_symbol("demo_pursuit", "MissingPolicy")


def test_research_package_statically_forbids_root_asset_imports(tmp_path):
    research_root = ROOT / "pursuit_evasion_rl" / "research"
    assert find_forbidden_research_imports(research_root) == ()
    assert_research_import_boundary(research_root)

    injected = tmp_path / "research"
    injected.mkdir()
    (injected / "bad.py").write_text(
        "from demo_pursuit import GoalEvader\nimport eval_trained\n",
        encoding="utf-8",
    )
    violations = find_forbidden_research_imports(injected)
    assert [(item.line, item.module) for item in violations] == [
        (1, "demo_pursuit"),
        (2, "eval_trained"),
    ]
    with pytest.raises(ResearchValidationError) as raised:
        assert_research_import_boundary(injected)
    assert raised.value.code == "FORBIDDEN_ROOT_IMPORT"


def test_staged_dispatch_preserves_argv_and_does_not_claim_missing_trainer():
    contract = resolve_script_shim("train_osm_pursuit")
    actually_available = importlib.util.find_spec(contract.package_module) is not None
    assert contract.package_entrypoint_available is actually_available
    assert contract.forwards_argv_unchanged

    if actually_available:
        assert contract.stage == "package_entrypoint"
        pytest.skip("real package trainer entry point is now installed")

    seen = []

    def legacy_main(argv):
        seen.append(argv)
        return 17

    argv = ["--episodes", "3", "--resume", "legacy.pt"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = dispatch_legacy_script(
            "train_osm_pursuit", argv, legacy_main=legacy_main
        )

    assert result == 17
    assert seen == [argv]
    assert any(item.category is DeprecationWarning for item in caught)
    assert contract.stage == "legacy_fallback"


def test_root_train_and_evaluate_scripts_are_thin_staged_wrappers():
    for filename, script_name in (
        ("train_osm_pursuit.py", "train_osm_pursuit"),
        ("eval_trained.py", "eval_trained"),
    ):
        source = (ROOT / filename).read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        assert any(
            isinstance(call.func, ast.Name)
            and call.func.id == "dispatch_legacy_script"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value == script_name
            for call in calls
        )
        assert "def _legacy_main(argv=None):" in source
        assert "parse_args(argv)" in source


def test_invalid_or_unpreservable_legacy_inputs_fail_closed():
    with pytest.raises(ResearchValidationError) as unsupported:
        convert_legacy_cli_arguments("unknown.py", {})
    assert unsupported.value.code == "UNSUPPORTED_LEGACY_SCRIPT"

    payload = {"police_actor": {"bad": object()}}
    with pytest.raises(ResearchValidationError) as unpreservable:
        convert_legacy_checkpoint(payload, source_content_hash="0" * 64)
    assert unpreservable.value.code == "UNSUPPORTED_CHECKPOINT_VALUE"
