"""Task 4.7: static architecture guard against legacy unmasked PPO usage.

Correctness Property 20 (action-mask support is identical during sampling and
PPO recomputation) is only a runtime guarantee if the PPO recomputation code
path can never reach a mask *recomputed* from the environment/network instead
of the one sealed at rollout time, and can never fall through to the legacy,
unmasked ``_ppo_update`` implementations in ``pursuit_evasion_rl/training/``.
This is an AST/import-level check, not a behavioral one: it inspects source
text without importing or executing it.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.offline

REPO_ROOT = Path(__file__).resolve().parents[2]
MASKED_MAPPO_PATH = REPO_ROOT / "pursuit_evasion_rl/research/policies/masked_mappo.py"
RESEARCH_ROOT = REPO_ROOT / "pursuit_evasion_rl/research"

# The legacy, unmasked PPO implementations this architecture must never reach.
_LEGACY_PPO_MODULES = (
    "pursuit_evasion_rl.training.algorithms",
    "pursuit_evasion_rl.training.gaussian_mappo",
)
_LEGACY_PPO_SYMBOL = "_ppo_update"

# The environment/mask-recomputation API the PPO recomputation path (owned by
# masked_mappo.py) must never import -- it has no environment-state mask API
# by construction (see the module's own docstring).
_ENVIRONMENT_MASK_MODULE_PREFIX = "pursuit_evasion_rl.osm_demo"


def _imported_module_names(tree: ast.AST) -> tuple[str, ...]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return tuple(names)


def _referenced_identifiers(tree: ast.AST) -> tuple[str, ...]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.alias):
            names.append(node.asname or node.name.rsplit(".", 1)[-1])
    return tuple(names)


def find_legacy_ppo_violations(source: str, *, path: str = "<source>") -> tuple[str, ...]:
    """Detect any reference to the legacy unmasked PPO modules or symbol."""
    tree = ast.parse(source, filename=path)
    violations: list[str] = []
    for module in _imported_module_names(tree):
        if any(module == legacy or module.startswith(f"{legacy}.") for legacy in _LEGACY_PPO_MODULES):
            violations.append(f"{path}: imports legacy PPO module {module!r}")
    if _LEGACY_PPO_SYMBOL in _referenced_identifiers(tree):
        violations.append(f"{path}: references the legacy {_LEGACY_PPO_SYMBOL!r} symbol")
    return tuple(violations)


def find_environment_mask_recomputation_violations(source: str, *, path: str = "<source>") -> tuple[str, ...]:
    """Detect any import of the environment/network mask-recomputation API."""
    tree = ast.parse(source, filename=path)
    violations: list[str] = []
    for module in _imported_module_names(tree):
        if module == _ENVIRONMENT_MASK_MODULE_PREFIX or module.startswith(f"{_ENVIRONMENT_MASK_MODULE_PREFIX}."):
            violations.append(f"{path}: imports the environment mask API via {module!r}")
    return tuple(violations)


def _research_package_files() -> tuple[Path, ...]:
    return tuple(sorted(RESEARCH_ROOT.rglob("*.py")))


def test_masked_mappo_module_has_no_environment_mask_api():
    source = MASKED_MAPPO_PATH.read_text(encoding="utf-8")
    violations = find_environment_mask_recomputation_violations(source, path=str(MASKED_MAPPO_PATH))
    assert violations == (), f"PPO recomputation module must never import the environment mask API: {violations}"


def test_masked_mappo_module_never_references_legacy_ppo():
    source = MASKED_MAPPO_PATH.read_text(encoding="utf-8")
    violations = find_legacy_ppo_violations(source, path=str(MASKED_MAPPO_PATH))
    assert violations == (), f"PPO recomputation module must never reach legacy unmasked PPO: {violations}"


def test_research_package_never_references_legacy_ppo():
    all_violations: list[str] = []
    for path in _research_package_files():
        source = path.read_text(encoding="utf-8")
        all_violations.extend(find_legacy_ppo_violations(source, path=str(path.relative_to(REPO_ROOT))))
    assert all_violations == [], f"legacy unmasked PPO references found in the research package: {all_violations}"


def test_checker_detects_an_injected_legacy_ppo_import():
    fixture_source = (
        "from pursuit_evasion_rl.training.algorithms import PPOAlgorithm\n"
        "\n"
        "def train():\n"
        "    return PPOAlgorithm()\n"
    )
    violations = find_legacy_ppo_violations(fixture_source, path="<injected fixture>")
    assert len(violations) == 1
    assert "pursuit_evasion_rl.training.algorithms" in violations[0]


def test_checker_detects_an_injected_legacy_ppo_symbol_reference():
    fixture_source = (
        "class Trainer:\n"
        "    def step(self, batch):\n"
        "        return self._ppo_update(batch)\n"
    )
    violations = find_legacy_ppo_violations(fixture_source, path="<injected fixture>")
    assert len(violations) == 1
    assert "_ppo_update" in violations[0]


def test_checker_detects_an_injected_environment_mask_import():
    fixture_source = (
        "from pursuit_evasion_rl.osm_demo.policies import build_action_mask\n"
        "\n"
        "def update(batch):\n"
        "    return build_action_mask(batch.network, 0)\n"
    )
    violations = find_environment_mask_recomputation_violations(fixture_source, path="<injected fixture>")
    assert len(violations) == 1
    assert "pursuit_evasion_rl.osm_demo.policies" in violations[0]


def test_checker_reports_no_violations_for_clean_fixture_source():
    fixture_source = (
        "import torch\n"
        "\n"
        "def update(batch):\n"
        "    return torch.zeros(1)\n"
    )
    assert find_legacy_ppo_violations(fixture_source) == ()
    assert find_environment_mask_recomputation_violations(fixture_source) == ()
