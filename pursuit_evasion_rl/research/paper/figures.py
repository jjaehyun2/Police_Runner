"""Claim-gated result figures and representative-trajectory selection (Requirements 14.9, 16.5-16.6).

A :class:`FigurePanel` carries the same Requirement 14.9 sidecar a table cell does
-- run identifier, generating function, Code_Revision, config Content_Hash, input
Content_Hashes and its own output Content_Hash -- and the same replay check
refuses a panel whose registered hash no longer follows from its recorded inputs
(Requirement 14.11-14.13).  A panel also cannot be constructed without an
eligible :class:`~pursuit_evasion_rl.research.claims.gate.ClaimGateReport`, so a
figure drawn straight from raw evaluation output is not representable.

Requirement 16.5 fixes what a result figure must contain: split panels, learning
curves, seed variation, behavior metrics, and one representative success and
failure trajectory.  :class:`ResultFigure` refuses to assemble unless every one of
:data:`REQUIRED_PANEL_KINDS` is present.

Requirement 16.6 -- the rule for picking those representative trajectories must be
recorded *before* anyone sees the result labels -- is enforced structurally rather
than by a promise:

* a :class:`TrajectorySelectionRule` is produced by
  :func:`freeze_trajectory_selection_rule`, which refuses an unsealed
  :class:`~pursuit_evasion_rl.research.protocol.ProtocolRecord` and stamps the
  sealed ``protocol_hash`` onto the rule.  A seal is what makes the rule
  demonstrably older than any result;
* :func:`select_representative_trajectory` takes the rule as a required argument
  and reads only ``rule.metric`` and ``rule.tie_break`` off each candidate.  A
  :class:`TrajectoryCandidate` carries no outcome label at all -- success and
  failure are two separately supplied pools -- so the selector has nothing to
  rationalize a rule from, and no constructor anywhere accepts candidates as an
  input to a rule;
* the bundle then checks the rule's ``protocol_hash`` against the manuscript's own
  sealed protocol, so a rule frozen under some other protocol is caught at export.

The residual hole is a caller hand-writing a ``TrajectorySelectionRule`` with a
fabricated ``protocol_hash``; that is exactly what the bundle-level check closes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from ..canonical import content_hash
from ..domain import exactly_one
from ..errors import ErrorRecord, ResearchValidationError
from ..protocol import ProtocolRecord
from .tables import (
    PAPER_ARTIFACT_SCHEMA_VERSION,
    PAPER_GENERATOR_CODE_REVISION,
    artifact_output_hash,
    record_cell_provenance,
    require_eligible_gate,
)

if TYPE_CHECKING:  # pragma: no cover - see tables._gate_report_type for the cycle
    from ..claims.gate import ClaimGateReport


def _fail(code: str, message: str, **kwargs: Any) -> None:
    raise ResearchValidationError(code, message, **kwargs)


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("MISSING_REQUIRED_FIELD", f"{name} must be a non-empty string", path=name, actual=value)
    return value


class FigurePanelKind(str, Enum):
    """The panel kinds Requirement 16.5 makes mandatory."""

    SPLIT = "split"
    LEARNING_CURVE = "learning_curve"
    SEED_VARIATION = "seed_variation"
    BEHAVIOR_METRIC = "behavior_metric"
    SUCCESS_TRAJECTORY = "success_trajectory"
    FAILURE_TRAJECTORY = "failure_trajectory"


#: Requirement 16.5: a result figure is incomplete without every one of these.
REQUIRED_PANEL_KINDS: tuple[FigurePanelKind, ...] = tuple(FigurePanelKind)

#: The two panel kinds whose content is chosen by a pre-registered rule (16.6).
TRAJECTORY_PANEL_KINDS: tuple[FigurePanelKind, ...] = (
    FigurePanelKind.SUCCESS_TRAJECTORY,
    FigurePanelKind.FAILURE_TRAJECTORY,
)


class TrajectoryOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


#: Which panel a selected trajectory belongs in.
TRAJECTORY_PANEL_FOR: Mapping[TrajectoryOutcome, FigurePanelKind] = {
    TrajectoryOutcome.SUCCESS: FigurePanelKind.SUCCESS_TRAJECTORY,
    TrajectoryOutcome.FAILURE: FigurePanelKind.FAILURE_TRAJECTORY,
}


# ---------------------------------------------------------------------------
# Requirement 14.9: the figure panel sidecar
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class FigurePanelProvenance:
    """One figure panel's Requirement 14.9 sidecar, shaped exactly like a cell's.

    As with :class:`~pursuit_evasion_rl.research.paper.tables.TableCellProvenance`,
    ``output_hash`` is recorded rather than verified at construction; the replay
    check lives in :func:`verify_panel_replay` and in the bundle's export gate.
    """

    artifact_id: str
    run_id: str
    generator: str
    code_revision: str = PAPER_GENERATOR_CODE_REVISION
    config_hash: str
    input_hashes: Mapping[str, str]
    output_hash: str
    schema_version: str = PAPER_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("artifact_id", "run_id", "generator", "code_revision", "config_hash", "output_hash"):
            _required_text(getattr(self, name), name)
        if not isinstance(self.input_hashes, Mapping) or not self.input_hashes:
            _fail(
                "MISSING_INPUT_HASHES",
                "input_hashes must map at least one input name to its content hash",
                path="input_hashes",
            )
        object.__setattr__(self, "input_hashes", dict(self.input_hashes))

    def replayed_output_hash(self, payload: Any) -> str:
        return artifact_output_hash(
            artifact_id=self.artifact_id,
            run_id=self.run_id,
            generator=self.generator,
            code_revision=self.code_revision,
            config_hash=self.config_hash,
            input_hashes=self.input_hashes,
            value=payload,
        )

    @property
    def provenance_hash(self) -> str:
        return content_hash(self)


def record_panel_provenance(
    *,
    artifact_id: str,
    run_id: str,
    generator: str,
    config_hash: str,
    input_hashes: Mapping[str, str],
    payload: Any,
    code_revision: str = PAPER_GENERATOR_CODE_REVISION,
) -> FigurePanelProvenance:
    """Build a panel sidecar whose ``output_hash`` follows from its recorded inputs."""
    cell = record_cell_provenance(
        artifact_id=artifact_id,
        run_id=run_id,
        generator=generator,
        config_hash=config_hash,
        input_hashes=input_hashes,
        value=payload,
        code_revision=code_revision,
    )
    return FigurePanelProvenance(
        artifact_id=cell.artifact_id,
        run_id=cell.run_id,
        generator=cell.generator,
        code_revision=cell.code_revision,
        config_hash=cell.config_hash,
        input_hashes=cell.input_hashes,
        output_hash=cell.output_hash,
    )


# ---------------------------------------------------------------------------
# Requirement 16.6: the pre-registered representative-trajectory rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class TrajectorySelectionRule:
    """The metric and tie-break ordering that pick a representative trajectory.

    Build this with :func:`freeze_trajectory_selection_rule`; ``protocol_hash``
    binds it to a sealed Research_Protocol, and the bundle refuses a rule bound to
    anything but the manuscript's own sealed protocol.
    """

    rule_id: str
    metric: str
    tie_break: tuple[str, ...]
    higher_is_better: bool
    protocol_hash: str
    declared_at_utc: str
    schema_version: str = PAPER_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("rule_id", "metric", "protocol_hash", "declared_at_utc"):
            _required_text(getattr(self, name), name)
        if not isinstance(self.higher_is_better, bool):
            _fail(
                "INVALID_SELECTION_RULE",
                "higher_is_better must be a boolean",
                path="higher_is_better",
                actual=self.higher_is_better,
            )
        tie_break = tuple(str(item) for item in self.tie_break)
        if not tie_break:
            _fail(
                "MISSING_TIE_BREAK_ORDERING",
                "Requirement 16.6 requires a recorded tie-break ordering, not just a metric",
                path="tie_break",
            )
        if len(set(tie_break)) != len(tie_break):
            _fail(
                "INVALID_SELECTION_RULE",
                "the tie-break ordering must not repeat a key",
                path="tie_break",
                actual=list(tie_break),
            )
        object.__setattr__(self, "tie_break", tie_break)

    @property
    def required_keys(self) -> tuple[str, ...]:
        return (self.metric, *self.tie_break)

    @property
    def rule_hash(self) -> str:
        return content_hash(self)


def freeze_trajectory_selection_rule(
    protocol: ProtocolRecord,
    *,
    rule_id: str,
    metric: str,
    tie_break: Sequence[str],
    higher_is_better: bool,
    declared_at_utc: str,
) -> TrajectorySelectionRule:
    """Record the rule against a sealed protocol, which is what dates it (16.6).

    An unsealed protocol is still editable, so a rule attached to one proves
    nothing about when it was written; that is refused here rather than at export.
    """
    if not isinstance(protocol, ProtocolRecord):
        _fail(
            "MISSING_PROTOCOL_RECORD",
            "a trajectory selection rule must be recorded against a Research_Protocol",
            path="protocol",
            actual=type(protocol).__name__,
        )
    if not protocol.is_sealed:
        _fail(
            "TRAJECTORY_RULE_NOT_PRE_REGISTERED",
            "the representative-trajectory rule must be recorded in a sealed protocol",
            path="protocol.is_sealed",
            expected=True,
            actual=False,
            details={"protocol_id": protocol.protocol_id},
        )
    return TrajectorySelectionRule(
        rule_id=rule_id,
        metric=metric,
        tie_break=tuple(tie_break),
        higher_is_better=higher_is_better,
        protocol_hash=protocol.protocol_hash,
        declared_at_utc=declared_at_utc,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class TrajectoryCandidate:
    """One candidate trajectory, carrying metrics and deliberately no outcome label.

    Which pool a candidate is passed in is the only thing that says whether it was
    a success or a failure, so the selector cannot sort on the label it is meant to
    be blind to.
    """

    trajectory_id: str
    run_id: str
    metrics: Mapping[str, float]
    schema_version: str = PAPER_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("trajectory_id", "run_id"):
            _required_text(getattr(self, name), name)
        if not isinstance(self.metrics, Mapping) or not self.metrics:
            _fail(
                "MISSING_TRAJECTORY_METRICS",
                "a trajectory candidate must carry the metrics a rule may select on",
                path="metrics",
            )
        object.__setattr__(self, "metrics", {str(key): float(item) for key, item in self.metrics.items()})


@dataclass(frozen=True, slots=True, kw_only=True)
class TrajectorySelection:
    """The chosen trajectory plus the rule and pool it was chosen from."""

    outcome: TrajectoryOutcome
    rule: TrajectorySelectionRule
    selected: TrajectoryCandidate
    considered_ids: tuple[str, ...]
    schema_version: str = PAPER_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", exactly_one(self.outcome, TrajectoryOutcome, path="outcome"))
        object.__setattr__(self, "considered_ids", tuple(str(item) for item in self.considered_ids))

    @property
    def panel_kind(self) -> FigurePanelKind:
        return TRAJECTORY_PANEL_FOR[self.outcome]


def select_representative_trajectory(
    rule: TrajectorySelectionRule,
    candidates: Sequence[TrajectoryCandidate] | Iterable[TrajectoryCandidate],
    *,
    outcome: TrajectoryOutcome,
) -> TrajectorySelection:
    """Pick one trajectory deterministically under a pre-recorded rule (16.6).

    The rule is the first parameter and there is no overload that derives one from
    ``candidates``; ordering reads ``rule.metric`` first, then each ``tie_break``
    key in the recorded order, and finally ``trajectory_id`` so the outcome is
    total even under exact ties.
    """
    if not isinstance(rule, TrajectorySelectionRule):
        _fail(
            "MISSING_SELECTION_RULE",
            "a representative trajectory requires its pre-registered selection rule",
            path="rule",
            actual=type(rule).__name__,
        )
    resolved = exactly_one(outcome, TrajectoryOutcome, path="outcome")
    pool = tuple(candidates)
    if not pool:
        _fail(
            "EMPTY_TRAJECTORY_POOL",
            "a representative trajectory cannot be selected from an empty pool",
            path="candidates",
            actual=resolved.value,
        )
    for candidate in pool:
        if not isinstance(candidate, TrajectoryCandidate):
            _fail("INVALID_TRAJECTORY_CANDIDATE", "candidates must be TrajectoryCandidate instances", path="candidates")
        missing = [key for key in rule.required_keys if key not in candidate.metrics]
        if missing:
            _fail(
                "TRAJECTORY_METRIC_NOT_RECORDED",
                "a candidate does not carry every metric the pre-registered rule selects on",
                path="candidates.metrics",
                expected=list(rule.required_keys),
                actual=missing,
                details={"trajectory_id": candidate.trajectory_id},
            )

    direction = -1.0 if rule.higher_is_better else 1.0

    def _order(candidate: TrajectoryCandidate) -> tuple[Any, ...]:
        return (
            direction * candidate.metrics[rule.metric],
            *(candidate.metrics[key] for key in rule.tie_break),
            candidate.trajectory_id,
        )

    selected = min(pool, key=_order)
    return TrajectorySelection(
        outcome=resolved,
        rule=rule,
        selected=selected,
        considered_ids=tuple(sorted(item.trajectory_id for item in pool)),
    )


# ---------------------------------------------------------------------------
# Requirement 16.5: the result figure
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class FigurePanel:
    """One panel, its sidecar, and the gate verdict its data descends from."""

    figure_id: str
    panel_key: str
    kind: FigurePanelKind
    payload: Any
    provenance: FigurePanelProvenance
    gate: "ClaimGateReport"
    selection: TrajectorySelection | None = None
    schema_version: str = PAPER_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("figure_id", "panel_key"):
            _required_text(getattr(self, name), name)
        object.__setattr__(self, "kind", exactly_one(self.kind, FigurePanelKind, path="kind"))
        if self.payload is None:
            _fail("MISSING_PANEL_PAYLOAD", "a figure panel must carry the data it plots", path="payload")
        if not isinstance(self.provenance, FigurePanelProvenance):
            _fail("MISSING_PANEL_PROVENANCE", "a figure panel requires its provenance sidecar", path="provenance")
        object.__setattr__(self, "gate", require_eligible_gate(self.gate, "gate"))
        if self.provenance.artifact_id != self.panel_id:
            _fail(
                "PANEL_PROVENANCE_IDENTIFIER_MISMATCH",
                "the sidecar must be registered under the panel it describes",
                path="provenance.artifact_id",
                expected=self.panel_id,
                actual=self.provenance.artifact_id,
            )
        if self.kind in TRAJECTORY_PANEL_KINDS:
            if not isinstance(self.selection, TrajectorySelection):
                _fail(
                    "MISSING_TRAJECTORY_SELECTION_RULE",
                    "a representative trajectory panel requires the rule that selected it",
                    path="selection",
                    actual=type(self.selection).__name__,
                )
            if self.selection.panel_kind is not self.kind:
                _fail(
                    "TRAJECTORY_OUTCOME_PANEL_MISMATCH",
                    "the selected trajectory outcome does not match the panel it is placed in",
                    path="selection.outcome",
                    expected=self.kind.value,
                    actual=self.selection.panel_kind.value,
                )
        elif self.selection is not None:
            _fail(
                "UNEXPECTED_TRAJECTORY_SELECTION",
                "only a representative trajectory panel carries a selection rule",
                path="selection",
                actual=self.kind.value,
            )

    @property
    def panel_id(self) -> str:
        return f"{self.figure_id}::{self.panel_key}"


def verify_panel_replay(panel: FigurePanel) -> tuple[ErrorRecord, ...]:
    """Requirement 14.11-14.13: recompute the panel hash from the recorded sidecar."""
    replayed = panel.provenance.replayed_output_hash(panel.payload)
    if replayed != panel.provenance.output_hash:
        return (
            ErrorRecord(
                code="PANEL_HASH_REPLAY_MISMATCH",
                message="replaying the recorded inputs does not reproduce the registered panel hash",
                path="provenance.output_hash",
                expected=replayed,
                actual=panel.provenance.output_hash,
                details={"panel_id": panel.panel_id, "run_id": panel.provenance.run_id},
            ),
        )
    return ()


def require_panel_replay(panel: FigurePanel) -> None:
    errors = verify_panel_replay(panel)
    if errors:
        error = errors[0]
        _fail(error.code, error.message, path=error.path, expected=error.expected, actual=error.actual)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResultFigure:
    """A gated result figure carrying every panel Requirement 16.5 mandates."""

    figure_id: str
    caption: str
    panels: tuple[FigurePanel, ...]
    schema_version: str = PAPER_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("figure_id", "caption"):
            _required_text(getattr(self, name), name)
        panels = tuple(self.panels)
        if not panels:
            _fail("EMPTY_RESULT_FIGURE", "a result figure requires at least one panel", path="panels")
        seen: set[str] = set()
        for panel in panels:
            if not isinstance(panel, FigurePanel):
                _fail("INVALID_FIGURE_PANEL", "panels must be FigurePanel instances", path="panels")
            if panel.figure_id != self.figure_id:
                _fail(
                    "FIGURE_PANEL_IDENTIFIER_MISMATCH",
                    "a panel belongs to a different figure",
                    path="panels.figure_id",
                    expected=self.figure_id,
                    actual=panel.figure_id,
                )
            if panel.panel_id in seen:
                _fail("DUPLICATE_FIGURE_PANEL", "figure panels must be unique", path="panels", actual=panel.panel_id)
            seen.add(panel.panel_id)
        kinds = {panel.kind for panel in panels}
        missing = [kind.value for kind in REQUIRED_PANEL_KINDS if kind not in kinds]
        if missing:
            _fail(
                "INCOMPLETE_RESULT_FIGURE",
                "Requirement 16.5 requires split, learning curve, seed variation, behavior metric "
                "and representative success and failure trajectory panels",
                path="panels",
                expected=[kind.value for kind in REQUIRED_PANEL_KINDS],
                actual=missing,
            )
        object.__setattr__(self, "panels", panels)

    @property
    def run_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(panel.provenance.run_id for panel in self.panels))

    @property
    def selection_rules(self) -> tuple[TrajectorySelectionRule, ...]:
        return tuple(panel.selection.rule for panel in self.panels if panel.selection is not None)

    @property
    def figure_hash(self) -> str:
        return content_hash(self)


def build_figure_panel(
    *,
    figure_id: str,
    panel_key: str,
    kind: FigurePanelKind,
    payload: Any,
    gate: "ClaimGateReport",
    run_id: str,
    config_hash: str,
    input_hashes: Mapping[str, str],
    selection: TrajectorySelection | None = None,
    generator: str = "pursuit_evasion_rl.research.paper.figures.build_figure_panel",
) -> FigurePanel:
    """Build a panel with a freshly computed sidecar, refusing an ungated payload."""
    require_eligible_gate(gate, "gate")
    panel_id = f"{figure_id}::{panel_key}"
    return FigurePanel(
        figure_id=figure_id,
        panel_key=panel_key,
        kind=kind,
        payload=payload,
        provenance=record_panel_provenance(
            artifact_id=panel_id,
            run_id=run_id,
            generator=generator,
            config_hash=config_hash,
            input_hashes=input_hashes,
            payload=payload,
        ),
        gate=gate,
        selection=selection,
    )


__all__ = (
    "REQUIRED_PANEL_KINDS",
    "TRAJECTORY_PANEL_FOR",
    "TRAJECTORY_PANEL_KINDS",
    "FigurePanel",
    "FigurePanelKind",
    "FigurePanelProvenance",
    "ResultFigure",
    "TrajectoryCandidate",
    "TrajectoryOutcome",
    "TrajectorySelection",
    "TrajectorySelectionRule",
    "build_figure_panel",
    "freeze_trajectory_selection_rule",
    "record_panel_provenance",
    "require_eligible_gate",
    "require_panel_replay",
    "select_representative_trajectory",
    "verify_panel_replay",
)
