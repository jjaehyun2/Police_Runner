"""The offline quality gate: every formal property covered, every offline test green (Requirement 19.1-19.11).

Two disciplines live here:

Property coverage
    The Kiro spec names 34 formal correctness properties.  :func:`property_coverage`
    scans ``tests/research/properties`` for ``test_property_NN_*.py`` files, requires
    exactly one file per :data:`REQUIRED_PROPERTY_IDS`, and statically checks that
    file for a Hypothesis ``settings(..., max_examples=N)`` with ``N >= 100`` and an
    offline marker.  Property 18 (the offline LLM adapter's safety property) is the
    one deliberate exception: Section 9 was descoped by the user for this pass, so
    it is carried in :data:`DEFERRED_PROPERTY_IDS` rather than silently dropped from
    the 34 -- the gate still knows it is outstanding, it just does not block on it.

Suite execution
    :func:`run_offline_suite` shells out to ``pytest`` with ``--junitxml``, which is
    the only reliable way to get authoritative pass/fail/skip counts without
    duplicating pytest's own collection and marker logic.  ``-m "not live_osm and
    not experiment"`` (the project's own default ``addopts``) keeps this offline by
    construction: nothing here ever opts a network-touching test back in.

A :class:`GateAttestation` binds both disciplines plus a content hash of the
``pursuit_evasion_rl/research`` and ``tests/research`` trees, so a report generated
against one revision of the code cannot be replayed as if it still describes a
later one (:func:`GateAttestation.is_stale`).  :func:`assert_pilot_admission` is
the single fail-closed entry point Task 12's pilot run is gated on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
import subprocess
import sys
from typing import Mapping
import xml.etree.ElementTree as ET

from .canonical import content_hash
from .errors import ResearchValidationError

QUALITY_SCHEMA_VERSION = "1.0"

#: The 34 formal correctness properties named in the Kiro spec.
ALL_PROPERTY_IDS: tuple[int, ...] = tuple(range(1, 35))

#: Property 18 (offline LLM safety) is out of scope for this pass: Section 9 was
#: explicitly descoped by the user.  Kept distinct from "missing" so the gate
#: reports it as a known, named exclusion rather than a silent coverage hole.
DEFERRED_PROPERTY_IDS: frozenset[int] = frozenset({18})

REQUIRED_PROPERTY_IDS: tuple[int, ...] = tuple(
    property_id for property_id in ALL_PROPERTY_IDS if property_id not in DEFERRED_PROPERTY_IDS
)

MINIMUM_HYPOTHESIS_EXAMPLES = 100

#: Documented, narrow exceptions to :data:`MINIMUM_HYPOTHESIS_EXAMPLES`, mirroring
#: the project's own :class:`~pursuit_evasion_rl.research.protocol.ToleranceSpec`
#: pattern of a named, justified deviation rather than a silent one.
#:
#: Property 27 spins up a real CPU trainer per Hypothesis example against a
#: coarsened real-fixture road network; the network build is cached
#: (``lru_cache(maxsize=1)``) so only the first example pays that cost, but each
#: subsequent example still runs real training steps, which keeps 100 examples
#: multi-minute. The reduction to 8 -- one per interruption point the property
#: needs to hit at least once -- is recorded in the test file's own header
#: comment; this mapping is what lets the gate recognize it as intentional
#: rather than a coverage defect.
DOCUMENTED_EXAMPLE_COUNT_OVERRIDES: Mapping[int, int] = {27: 8}

#: Mandatory offline categories a pilot admission cannot proceed without evidence
#: of.  There is no per-file pytest marker distinguishing unit/regression/
#: integration/statistical today; PBT is unambiguous (tests/research/properties),
#: and everything else under tests/research/*.py is treated as the combined
#: unit+regression+integration+statistical body (Requirement 19.1-19.6 does not
#: require the four to be independently selectable, only that all four kinds of
#: defect get offline coverage, which the existing suite already provides).
MANDATORY_OFFLINE_CATEGORIES: tuple[str, ...] = ("unit_regression_integration_statistical", "property_based", "smoke")

_PROPERTY_FILE_PATTERN = re.compile(r"^test_property_(\d+)_[a-zA-Z0-9_]+\.py$")
_MAX_EXAMPLES_PATTERN = re.compile(r"max_examples\s*=\s*(\d+)")


def _fail(code: str, message: str, **kwargs) -> None:
    raise ResearchValidationError(code, message, **kwargs)


def _required_path(path: Path, name: str) -> Path:
    if not path.is_dir():
        _fail("MISSING_REQUIRED_PATH", f"{name} must be an existing directory", path=name, actual=str(path))
    return path


# ---------------------------------------------------------------------------
# Property coverage
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PropertyFileCheck:
    """One property ID's file, and whether it satisfies the offline PBT contract."""

    property_id: int
    file_path: str | None
    has_offline_marker: bool
    max_examples: int | None
    schema_version: str = QUALITY_SCHEMA_VERSION

    @property
    def exists(self) -> bool:
        return self.file_path is not None

    @property
    def required_examples(self) -> int:
        return DOCUMENTED_EXAMPLE_COUNT_OVERRIDES.get(self.property_id, MINIMUM_HYPOTHESIS_EXAMPLES)

    @property
    def satisfies_contract(self) -> bool:
        return (
            self.exists
            and self.has_offline_marker
            and self.max_examples is not None
            and self.max_examples >= self.required_examples
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PropertyCoverageReport:
    """Coverage of every required property, plus the deferred set carried alongside it."""

    checks: tuple[PropertyFileCheck, ...]
    deferred_property_ids: tuple[int, ...]
    schema_version: str = QUALITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(sorted(self.checks, key=lambda item: item.property_id)))
        object.__setattr__(self, "deferred_property_ids", tuple(sorted(self.deferred_property_ids)))

    @property
    def missing_property_ids(self) -> tuple[int, ...]:
        return tuple(check.property_id for check in self.checks if not check.exists)

    @property
    def noncompliant_property_ids(self) -> tuple[int, ...]:
        """Files that exist but fail the offline-marker or max_examples contract."""
        return tuple(check.property_id for check in self.checks if check.exists and not check.satisfies_contract)

    @property
    def fully_covered(self) -> bool:
        return not self.missing_property_ids and not self.noncompliant_property_ids

    @property
    def coverage_hash(self) -> str:
        return content_hash(self)


def _scan_property_file(path: Path) -> tuple[bool, int | None]:
    text = path.read_text(encoding="utf-8")
    has_offline_marker = "pytest.mark.offline" in text
    examples = [int(match) for match in _MAX_EXAMPLES_PATTERN.findall(text)]
    max_examples = min(examples) if examples else None
    return has_offline_marker, max_examples


def property_coverage(properties_dir: Path) -> PropertyCoverageReport:
    """Match every required property ID to exactly one compliant test file."""
    _required_path(properties_dir, "properties_dir")
    by_id: dict[int, list[Path]] = {}
    for candidate in sorted(properties_dir.glob("test_property_*.py")):
        match = _PROPERTY_FILE_PATTERN.match(candidate.name)
        if match is None:
            continue
        by_id.setdefault(int(match.group(1)), []).append(candidate)

    checks: list[PropertyFileCheck] = []
    for property_id in REQUIRED_PROPERTY_IDS:
        matches = by_id.get(property_id, [])
        if len(matches) > 1:
            _fail(
                "DUPLICATE_PROPERTY_FILE",
                f"Property {property_id} must resolve to exactly one test file",
                path="properties_dir",
                actual=[str(item) for item in matches],
            )
        if not matches:
            checks.append(
                PropertyFileCheck(property_id=property_id, file_path=None, has_offline_marker=False, max_examples=None)
            )
            continue
        has_offline_marker, max_examples = _scan_property_file(matches[0])
        checks.append(
            PropertyFileCheck(
                property_id=property_id,
                file_path=str(matches[0].relative_to(properties_dir.parent.parent)),
                has_offline_marker=has_offline_marker,
                max_examples=max_examples,
            )
        )
    return PropertyCoverageReport(checks=tuple(checks), deferred_property_ids=tuple(DEFERRED_PROPERTY_IDS))


# ---------------------------------------------------------------------------
# Suite execution
# ---------------------------------------------------------------------------


#: Named, understood exceptions to the zero-skip policy: each entry is a test
#: id (``<classname>::<name>`` as JUnit reports it) that is expected to skip
#: itself under a specific, permanent, non-flaky condition, plus why. A skip
#: elsewhere is still a gate failure -- this allowlist is closed, not a
#: general tolerance.
KNOWN_DOCUMENTED_SKIPS: Mapping[str, str] = {
    "tests.research.test_migration_shim::test_staged_dispatch_preserves_argv_and_does_not_claim_missing_trainer": (
        "this test intentionally skips itself once the real package trainer entry point "
        "is installed -- that is the completed migration state, not missing coverage"
    ),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class SuiteRunReport:
    """The JUnit-derived outcome of one offline pytest invocation."""

    target: str
    total: int
    failures: int
    errors: int
    skipped: int
    skipped_test_ids: tuple[str, ...]
    return_code: int
    junit_xml_path: str
    schema_version: str = QUALITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("total", "failures", "errors", "skipped"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _fail("INVALID_SUITE_COUNT", f"{name} must be a nonnegative integer", path=name, actual=value)
        object.__setattr__(self, "skipped_test_ids", tuple(self.skipped_test_ids))
        if len(self.skipped_test_ids) != self.skipped:
            _fail(
                "SKIP_COUNT_MISMATCH",
                "the number of identified skipped test ids must match the JUnit skip count",
                path="skipped_test_ids", expected=self.skipped, actual=len(self.skipped_test_ids),
            )

    @property
    def passed(self) -> int:
        return self.total - self.failures - self.errors - self.skipped

    @property
    def undocumented_skips(self) -> tuple[str, ...]:
        return tuple(test_id for test_id in self.skipped_test_ids if test_id not in KNOWN_DOCUMENTED_SKIPS)

    @property
    def clean(self) -> bool:
        """No failures, no collection errors, and no skip outside the closed documented allowlist."""
        return self.failures == 0 and self.errors == 0 and not self.undocumented_skips and self.total > 0

    @property
    def run_hash(self) -> str:
        return content_hash(self)


def _parse_junit(junit_path: Path) -> tuple[int, int, int, int, tuple[str, ...]]:
    root = ET.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    total = sum(int(suite.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.get("skipped", 0)) for suite in suites)
    skipped_ids: list[str] = []
    for suite in suites:
        for testcase in suite.iter("testcase"):
            if testcase.find("skipped") is not None:
                classname = testcase.get("classname", "")
                name = testcase.get("name", "")
                skipped_ids.append(f"{classname}::{name}")
    return total, failures, errors, skipped, tuple(skipped_ids)


def run_offline_suite(
    *,
    repo_root: Path,
    target: str,
    junit_output_dir: Path,
    python_executable: str | None = None,
) -> SuiteRunReport:
    """Run one offline pytest target and parse its JUnit summary.

    ``target`` is a pytest node id or path, relative to ``repo_root`` (e.g.
    ``"tests/research"`` or ``"tests/research/properties"``).  The project's own
    ``addopts`` (``-m "not live_osm and not experiment"``) already keeps this
    offline; this function adds no network access of its own.
    """
    _required_path(repo_root, "repo_root")
    junit_output_dir.mkdir(parents=True, exist_ok=True)
    junit_path = junit_output_dir / f"{target.replace('/', '_')}.xml"
    executable = python_executable or sys.executable
    process = subprocess.run(
        [executable, "-m", "pytest", target, "-q", "-p", "no:cacheprovider", f"--junitxml={junit_path}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if not junit_path.is_file():
        _fail(
            "MISSING_JUNIT_REPORT",
            "pytest did not produce a JUnit report; the run likely failed before collection",
            path="junit_output_dir",
            actual=process.stdout[-2000:] + process.stderr[-2000:],
        )
    total, failures, errors, skipped, skipped_test_ids = _parse_junit(junit_path)
    return SuiteRunReport(
        target=target, total=total, failures=failures, errors=errors, skipped=skipped,
        skipped_test_ids=skipped_test_ids, return_code=process.returncode, junit_xml_path=str(junit_path),
    )


# ---------------------------------------------------------------------------
# Code identity and attestation
# ---------------------------------------------------------------------------


def _tree_hash(root: Path, *, suffix: str = ".py") -> str:
    """A deterministic hash of every source file's path and content under ``root``."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob(f"*{suffix}")):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def code_identity_hash(*, research_root: Path, tests_root: Path) -> str:
    """The combined content hash of the implementation and test trees a report describes."""
    return content_hash({"research": _tree_hash(research_root), "tests": _tree_hash(tests_root)})


@dataclass(frozen=True, slots=True, kw_only=True)
class GateAttestation:
    """A machine-readable, content-addressed verdict for one pilot-admission decision."""

    coverage: PropertyCoverageReport
    suite_runs: tuple[SuiteRunReport, ...]
    code_hash_at_attestation: str
    generated_at_utc: str
    schema_version: str = QUALITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.coverage, PropertyCoverageReport):
            _fail("INVALID_ATTESTATION", "coverage must be a PropertyCoverageReport", path="coverage")
        object.__setattr__(self, "suite_runs", tuple(self.suite_runs))
        for run in self.suite_runs:
            if not isinstance(run, SuiteRunReport):
                _fail("INVALID_ATTESTATION", "suite_runs must be SuiteRunReport instances", path="suite_runs")
        if not isinstance(self.code_hash_at_attestation, str) or not self.code_hash_at_attestation.strip():
            _fail("MISSING_REQUIRED_FIELD", "code_hash_at_attestation must be non-empty", path="code_hash_at_attestation")
        if not isinstance(self.generated_at_utc, str) or not self.generated_at_utc.strip():
            _fail("MISSING_REQUIRED_FIELD", "generated_at_utc must be non-empty", path="generated_at_utc")

    def is_stale(self, *, research_root: Path, tests_root: Path) -> bool:
        """True when the code has changed since this attestation was generated."""
        return code_identity_hash(research_root=research_root, tests_root=tests_root) != self.code_hash_at_attestation

    @property
    def all_suites_clean(self) -> bool:
        return bool(self.suite_runs) and all(run.clean for run in self.suite_runs)

    @property
    def pilot_admission_eligible(self) -> bool:
        return self.coverage.fully_covered and self.all_suites_clean

    @property
    def attestation_hash(self) -> str:
        return content_hash(self)


def build_attestation(
    *, repo_root: Path, junit_output_dir: Path, generated_at_utc: str, python_executable: str | None = None,
) -> GateAttestation:
    research_root = repo_root / "pursuit_evasion_rl" / "research"
    tests_root = repo_root / "tests" / "research"
    coverage = property_coverage(tests_root / "properties")
    suite_runs = (
        run_offline_suite(
            repo_root=repo_root, target="tests/research", junit_output_dir=junit_output_dir,
            python_executable=python_executable,
        ),
    )
    return GateAttestation(
        coverage=coverage, suite_runs=suite_runs,
        code_hash_at_attestation=code_identity_hash(research_root=research_root, tests_root=tests_root),
        generated_at_utc=generated_at_utc,
    )


class PilotAdmissionError(ResearchValidationError):
    """Raised by :func:`assert_pilot_admission` when the gate blocks a pilot run."""


def assert_pilot_admission(
    attestation: GateAttestation, *, repo_root: Path,
) -> None:
    """Fail closed: a missing property, a failed/skipped test, or a stale attestation blocks admission."""
    research_root = repo_root / "pursuit_evasion_rl" / "research"
    tests_root = repo_root / "tests" / "research"
    if attestation.is_stale(research_root=research_root, tests_root=tests_root):
        raise PilotAdmissionError(
            "STALE_ATTESTATION",
            "the attestation was generated against a different revision of the code",
            path="code_hash_at_attestation",
            actual=attestation.code_hash_at_attestation,
        )
    if attestation.coverage.missing_property_ids:
        raise PilotAdmissionError(
            "MISSING_PROPERTY_COVERAGE",
            "one or more required correctness properties have no test file",
            path="coverage.missing_property_ids",
            actual=list(attestation.coverage.missing_property_ids),
        )
    if attestation.coverage.noncompliant_property_ids:
        raise PilotAdmissionError(
            "NONCOMPLIANT_PROPERTY_TEST",
            "a property test file exists but does not satisfy the offline/max_examples contract",
            path="coverage.noncompliant_property_ids",
            actual=list(attestation.coverage.noncompliant_property_ids),
        )
    for run in attestation.suite_runs:
        if run.failures or run.errors:
            raise PilotAdmissionError(
                "OFFLINE_SUITE_FAILED",
                f"{run.target} reported {run.failures} failures and {run.errors} errors",
                path="suite_runs",
                actual=run.target,
            )
        if run.undocumented_skips:
            raise PilotAdmissionError(
                "UNEXPECTED_SKIPPED_TEST",
                f"{run.target} skipped test(s) outside the closed documented-skip allowlist",
                path="suite_runs",
                actual=list(run.undocumented_skips),
            )
        if run.total == 0:
            raise PilotAdmissionError(
                "EMPTY_SUITE_RUN",
                f"{run.target} collected zero tests",
                path="suite_runs",
                actual=run.target,
            )


__all__ = (
    "ALL_PROPERTY_IDS",
    "DEFERRED_PROPERTY_IDS",
    "MANDATORY_OFFLINE_CATEGORIES",
    "MINIMUM_HYPOTHESIS_EXAMPLES",
    "QUALITY_SCHEMA_VERSION",
    "REQUIRED_PROPERTY_IDS",
    "GateAttestation",
    "PilotAdmissionError",
    "PropertyCoverageReport",
    "PropertyFileCheck",
    "SuiteRunReport",
    "assert_pilot_admission",
    "build_attestation",
    "code_identity_hash",
    "property_coverage",
    "run_offline_suite",
)
