"""Property 28 coverage: every paper number and graphic has immutable run provenance."""

from __future__ import annotations

import dataclasses
from string import ascii_lowercase, digits, hexdigits

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.research.claims.gate import ClaimGateReport
from pursuit_evasion_rl.research.domain import ClaimScope, ClaimStatus
from pursuit_evasion_rl.research.errors import ErrorRecord, ResearchValidationError
from pursuit_evasion_rl.research.paper.figures import (
    FigurePanelProvenance,
    record_panel_provenance,
    verify_panel_replay,
)
from pursuit_evasion_rl.research.paper.figures import FigurePanelKind, build_figure_panel
from pursuit_evasion_rl.research.paper.tables import (
    ResultTableCell,
    TableCellProvenance,
    artifact_output_hash,
    record_cell_provenance,
    verify_cell_replay,
)
from pursuit_evasion_rl.research.execution import MeasuredResult

# **Property 28: Every paper number and graphic has immutable run provenance**
# **Validates: Requirements 14.9, 16.4, 16.10, 19.7-19.8**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)

_short = st.text(ascii_lowercase + digits, min_size=1, max_size=16)
_hex64 = st.text(hexdigits.lower(), min_size=64, max_size=64)


def _eligible_gate(claim_id: str) -> ClaimGateReport:
    return ClaimGateReport(
        claim_id=claim_id, claim_hash="a" * 64, scope=ClaimScope.RESULT,
        status=ClaimStatus.SUPPORTED, eligible=True, errors=(),
    )


@st.composite
def provenance_inputs(draw: st.DrawFn) -> dict:
    return {
        "artifact_id": draw(_short),
        "run_id": draw(_short),
        "generator": draw(_short),
        "config_hash": draw(_hex64),
        "input_hashes": {"metrics": draw(_hex64)},
        "value": draw(st.one_of(st.floats(allow_nan=False, allow_infinity=False), st.none(), _short)),
    }


@_PBT_SETTINGS
@given(fields=provenance_inputs())
def test_a_freshly_recorded_cell_sidecar_always_replays_cleanly(fields: dict) -> None:
    provenance = record_cell_provenance(**fields)
    assert provenance.output_hash == provenance.replayed_output_hash(fields["value"])
    assert provenance.output_hash == artifact_output_hash(
        artifact_id=fields["artifact_id"], run_id=fields["run_id"], generator=fields["generator"],
        code_revision=provenance.code_revision, config_hash=fields["config_hash"],
        input_hashes=fields["input_hashes"], value=fields["value"],
    )


@_PBT_SETTINGS
@given(
    fields=provenance_inputs(),
    mutate_field=st.sampled_from(("run_id", "generator", "config_hash", "value")),
    replacement=_short,
)
def test_mutating_any_recorded_input_or_the_value_breaks_the_replay(fields: dict, mutate_field: str, replacement: str) -> None:
    provenance = record_cell_provenance(**fields)
    mutated_value = fields["value"]
    mutated_fields = dict(fields)
    if mutate_field == "value":
        mutated_value = replacement if fields["value"] != replacement else replacement + "-x"
    else:
        original = fields[mutate_field]
        new_value = replacement if original != replacement else replacement + "-x"
        mutated_fields[mutate_field] = new_value
        provenance = dataclasses.replace(provenance, **{mutate_field: new_value})

    replayed = provenance.replayed_output_hash(mutated_value)
    assert replayed != provenance.output_hash


@_PBT_SETTINGS
@given(fields=provenance_inputs())
def test_input_hashes_cannot_be_empty_for_either_sidecar_kind(fields: dict) -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        TableCellProvenance(
            artifact_id=fields["artifact_id"], run_id=fields["run_id"], generator=fields["generator"],
            config_hash=fields["config_hash"], input_hashes={}, output_hash="a" * 64,
        )
    assert excinfo.value.code == "MISSING_INPUT_HASHES"

    with pytest.raises(ResearchValidationError) as excinfo_panel:
        FigurePanelProvenance(
            artifact_id=fields["artifact_id"], run_id=fields["run_id"], generator=fields["generator"],
            config_hash=fields["config_hash"], input_hashes={}, output_hash="a" * 64,
        )
    assert excinfo_panel.value.code == "MISSING_INPUT_HASHES"


@_PBT_SETTINGS
@given(fields=provenance_inputs())
def test_a_table_cell_cannot_be_constructed_without_an_eligible_gate(fields: dict) -> None:
    value = MeasuredResult(value=fields["value"] if not isinstance(fields["value"], str) else None)
    provenance = record_cell_provenance(**{**fields, "value": value})
    with pytest.raises(ResearchValidationError) as excinfo:
        ResultTableCell(
            table_id=fields["artifact_id"], row_key="row", column_key="col", value=value,
            provenance=dataclasses.replace(provenance, artifact_id=f"{fields['artifact_id']}::row::col"),
            gate=None, source_kind="test", source_id="src",
        )
    assert excinfo.value.code == "UNSOURCED_ARTIFACT_VALUE"

    rejected = ClaimGateReport(
        claim_id="claim-x", claim_hash="a" * 64, scope=ClaimScope.RESULT,
        status=ClaimStatus.FALSIFIED, eligible=False,
        errors=(ErrorRecord(code="X", message="m", path="p"),),
    )
    with pytest.raises(ResearchValidationError) as excinfo_rejected:
        ResultTableCell(
            table_id=fields["artifact_id"], row_key="row", column_key="col", value=value,
            provenance=dataclasses.replace(provenance, artifact_id=f"{fields['artifact_id']}::row::col"),
            gate=rejected, source_kind="test", source_id="src",
        )
    assert excinfo_rejected.value.code == "CLAIM_GATE_REJECTED_SOURCE"


@_PBT_SETTINGS
@given(fields=provenance_inputs())
def test_a_figure_panel_cannot_be_constructed_without_an_eligible_gate(fields: dict) -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        build_figure_panel(
            figure_id=fields["artifact_id"], panel_key="split", kind=FigurePanelKind.SPLIT,
            payload={"series": [1.0]}, gate=None, run_id=fields["run_id"],
            config_hash=fields["config_hash"], input_hashes=fields["input_hashes"],
        )
    assert excinfo.value.code == "UNSOURCED_ARTIFACT_VALUE"


@_PBT_SETTINGS
@given(fields=provenance_inputs())
def test_a_panel_whose_registered_hash_does_not_replay_is_rejected(fields: dict) -> None:
    gate = _eligible_gate(f"claim-{fields['artifact_id']}")
    payload = {"series": [1.0, 2.0]}
    panel = build_figure_panel(
        figure_id=fields["artifact_id"], panel_key="split", kind=FigurePanelKind.SPLIT,
        payload=payload, gate=gate, run_id=fields["run_id"], config_hash=fields["config_hash"],
        input_hashes=fields["input_hashes"],
    )
    assert verify_panel_replay(panel) == ()

    tampered = dataclasses.replace(
        panel, provenance=dataclasses.replace(panel.provenance, output_hash="f" * 64)
    )
    errors = verify_panel_replay(tampered)
    assert [item.code for item in errors] == ["PANEL_HASH_REPLAY_MISMATCH"]
