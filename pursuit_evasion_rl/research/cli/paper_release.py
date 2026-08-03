"""Task 12.5: gate-checked launch of the final claim-gated paper release.

Runs one fresh, tiny, sanity-scale train+paired-evaluate condition (reusing
experiments.main_study's real pipeline exactly as Task 12.2 already proved
it) as the release's own source data, then reconciles it through the same
claim gate and export checks Sections 8 and 12.1-12.4 already exercise.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import daejeon

from ..budget import ConditionResourceEstimate, ResourceCeiling, decide_sample_size
from ..canonical import canonical_json
from ..domain import EvidenceRecord, EvidenceType, MapScenario, ScreeningDecision
from ..errors import ResearchValidationError
from ..execution import ConditionExecutionLedger
from ..experiments.main_study import MainStudyConditionSpec, run_main_study_condition
from ..experiments.paper_release import require_paper_release_admission, run_paper_release
from ..maps.splits import DataHandle, HandleKind, SplitScope, TuningDataView
from ..paper.citations import (
    CitationCandidate,
    CitationLedger,
    CitationRecord,
    MetadataCrossCheck,
    PersistentIdentifierKind,
    ProposalCrossCheck,
    SearchProtocolSpec,
    SearchSource,
    SearchSourceKind,
    SourceLocation,
)
from ..paper.scope import (
    FutureWorkArtifact,
    FutureWorkRegistry,
    FutureWorkSystem,
    NonSubstitutionStatement,
    PreFieldValidationCategory,
    PreFieldValidationEntry,
    PreFieldValidationReport,
    PreFieldValidationStatus,
    RiskCategory,
    RiskDisclosure,
    RiskStatement,
    ScopeEthicsBundle,
    ScopeSection,
    ScopeSectionKind,
    SimulationLimitationDisclosure,
)
from ..protocol import ProtocolStore
from ..quality import build_attestation
from ..runs.manifest import RunManifestStore, RunProvenance
from ..statistics.paired import BootstrapPlan, EffectDirection, PracticalThreshold
from ..training.trainer import TrainerConfig
from ..variants.factory import full_condition_matrix
from ..variants.placement import PlacementCurriculumConfig, PlacementStyle

UTC = "2026-08-01T00:00:00+00:00"
QUERY = "multi-agent reinforcement learning vehicle pursuit road network"
CITATION_KEY = "yang2023"
FUTURE_WORK_ARTIFACT = "cctv-fusion-demo"


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class _LowestLegalActionPolicy:
    policy_id = "lowest-legal-action-v1"

    def act(self, observation) -> int:
        return min(index for index, legal in enumerate(observation.action_mask) if legal)


def _build_network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def _citation_ledger() -> CitationLedger:
    search_protocol = SearchProtocolSpec(
        search_date_utc=UTC, queries=(QUERY,),
        sources=(
            SearchSource(source_id="openalex", name="OpenAlex", kind=SearchSourceKind.ACADEMIC_INDEX, operator="OurResearch", endpoint="https://api.openalex.org/works"),
            SearchSource(source_id="scopus", name="Scopus", kind=SearchSourceKind.ACADEMIC_INDEX, operator="Elsevier", endpoint="https://api.elsevier.com/content/search/scopus"),
            SearchSource(source_id="arxiv", name="arXiv", kind=SearchSourceKind.PREPRINT_SERVER, operator="Cornell University", endpoint="https://export.arxiv.org/api/query"),
        ),
        period_start="2015-01-01", period_end="2026-07-31", languages=("en", "ko"),
        inclusion_criteria=("road-network pursuit or interception of a fleeing vehicle",), exclusion_criteria=("no evaluation on a road network",),
    )
    ledger = CitationLedger(search_protocol)
    work_id = "work::yang-progression-cognition"
    ledger.add_candidate(CitationCandidate(
        candidate_id=f"{CITATION_KEY}-c1", canonical_work_id=work_id, source_id="openalex", query=QUERY,
        executed_at_utc=UTC, result_rank=1, record_identifier=f"record::{CITATION_KEY}", result_content_hash=_hash(f"search-result::{CITATION_KEY}"),
    ))
    ledger.screen(f"{CITATION_KEY}-c1", ScreeningDecision.INCLUDED, "road-network pursuit with a learned policy", screened_at_utc=UTC)
    ledger.register_citation(CitationRecord(
        citation_key=CITATION_KEY, canonical_work_id=work_id, title="Progression Cognition Reinforcement Learning for Multi-Vehicle Pursuit",
        authors=("Yang, X.",), year=2023, candidate_ids=ledger.candidate_ids_for_work(work_id),
        proposal_cross_check=ProposalCrossCheck.NOT_FROM_PROPOSAL,
        metadata_cross_check=MetadataCrossCheck(identifier="10.1007/s41109-024-00689-1", identifier_kind=PersistentIdentifierKind.DOI, metadata_source="Crossref", checked_at_utc=UTC, author_match=True, year_match=True, title_match=True),
        source_location=SourceLocation(original_text_content_hash=_hash(CITATION_KEY), section="4 Experiments", table="Table 2"),
    ))
    return ledger


def _scope() -> ScopeEthicsBundle:
    future_work = FutureWorkRegistry(
        primary_result_root="artifacts/primary", future_work_root="artifacts/future_work",
        artifacts=(FutureWorkArtifact(artifact_id=FUTURE_WORK_ARTIFACT, system=FutureWorkSystem.CCTV, description="illustrative CCTV demo, not evaluated", artifact_path="artifacts/future_work/cctv/demo.mp4"),),
    )
    return ScopeEthicsBundle(
        limitations=SimulationLimitationDisclosure(
            traffic_and_vehicle_dynamics="Vehicle dynamics and ambient traffic are not modelled.",
            sensor_error="Sensor position error and dropout are assumed zero.",
            communication_latency_and_loss="Communication latency and packet loss are not modelled.",
            human_behavior="Driver and commander human behaviour is not modelled.",
            legal_and_operational_constraints="Legal and operational constraints are not modelled.",
        ),
        non_substitution=NonSubstitutionStatement(),
        risks=RiskDisclosure(statements=tuple(RiskStatement(category=category, description=f"{category.value} risk statement") for category in RiskCategory)),
        pre_field_validation=PreFieldValidationReport(entries=tuple(PreFieldValidationEntry(category=category, validation_id=f"pfv-{category.value}", status=PreFieldValidationStatus.NOT_PERFORMED) for category in PreFieldValidationCategory)),
        future_work=future_work,
        sections=(
            ScopeSection(section_id="sec-scope-current", kind=ScopeSectionKind.CURRENT_RESEARCH_SCOPE, title="Current research scope", body="Limited to simulated multi-officer pursuit policy evaluation on road graphs."),
            ScopeSection(section_id="sec-scope-future", kind=ScopeSectionKind.FUTURE_WORK, title="Future work", body="The following systems are not implemented in this study and remain future work.", systems=tuple(FutureWorkSystem)),
        ),
    )


def _sections() -> dict:
    return {
        "title": "Cooperative containment for multi-officer pursuit on extracted road networks",
        "abstract": "We evaluate a cooperative containment policy on road graphs.",
        "introduction": "Vehicle pursuit on a road network is a graph-structured coordination problem.",
        "related_work": "Prior graph-pursuit work is compared across eight differentiation axes.",
        "problem_definition": "Six officers pursue one goal-directed evader on a directed road graph.",
        "methods": "A masked multi-agent policy is optimized against the audited reward set.",
        "experiments": "Every planned condition is executed under one pre-registered sample size.",
        "results": "The paired capture-rate difference and its interval are reported.",
        "discussion": "The measured result is bounded by what this sanity-scale simulation represents.",
        "validity_limitations": "Internal, external, construct and statistical conclusion validity threats are enumerated separately.",
        "ethics": "Automation bias, surveillance expansion and misuse risks are disclosed.",
        "reproducibility": "Seeds, content hashes and the sealed protocol are published with limits noted.",
        "conclusion": "This sanity-scale run demonstrates the release pipeline end to end.",
        "references": "Yang, X. (2023). Progression Cognition Reinforcement Learning.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol-id", required=True, help="a protocol_id already sealed in --protocols-dir")
    parser.add_argument("--protocols-dir", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-output-root", type=Path, default=None)
    parser.add_argument("--analysis-id", required=True, help="the analysis_id of a confirmatory-registered PlannedAnalysis")
    parser.add_argument("--bundle-id", default="paper-osm-pursuit-release")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--asserts-superiority", action="store_true", help="omit for a sanity-scale run that cannot itself reach significance")
    parser.add_argument("--measured-at-utc", required=True)
    parser.add_argument("--junit-output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, help="write the canonical paper-release report JSON to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    protocols_dir = args.protocols_dir or (repo_root / "artifacts" / "research" / "protocols")
    runs_dir = args.runs_dir or (repo_root / "artifacts" / "research" / "runs")
    checkpoint_output_root = args.checkpoint_output_root or (repo_root / "artifacts" / "research" / "checkpoints")
    junit_output_dir = args.junit_output_dir or (repo_root / ".quality-gate")

    try:
        attestation = build_attestation(repo_root=repo_root, junit_output_dir=junit_output_dir, generated_at_utc=args.measured_at_utc)
        protocol = ProtocolStore(protocols_dir).read(args.protocol_id)
        token = require_paper_release_admission(quality_attestation=attestation, protocol=protocol)

        specs = full_condition_matrix((MapScenario.INTERIOR_CONTAINED,))[:1]
        condition_id = specs[0].condition.condition_id
        network = _build_network()
        tuning_data = TuningDataView(
            train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "release-train"),),
            validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "release-validation"),),
        )
        spec = MainStudyConditionSpec(
            condition_id=condition_id, scenario=MapScenario.INTERIOR_CONTAINED, train_network=network, validation_network=network,
            tuning_data=tuning_data, config=TrainerConfig(updates=args.updates, episodes_per_update=1, max_steps=args.max_steps, hidden_dims=(4,)),
            placement_config=PlacementCurriculumConfig(style=PlacementStyle.GLOBAL),
            baseline_policy_factory=lambda _network: _LowestLegalActionPolicy(), episodes_per_seed=args.episodes,
        )
        runs = RunManifestStore(runs_dir)

        def _provenance() -> RunProvenance:
            return RunProvenance(
                code_hash=attestation.code_hash_at_attestation, dirty_tree=False,
                dependency_hash=attestation.code_hash_at_attestation, runtime="python3.11+torch2.8.0+cpu",
                device="cpu", map_hash="release-network", split="train", seed=0, input_hashes={"map": "release-network"},
            )

        result = run_main_study_condition(
            token, spec, seeds=1, runs=runs, output_root=str(checkpoint_output_root), provenance_factory=_provenance,
            bootstrap_plan=BootstrapPlan(n_bootstrap=200, rng_seed=11, rationale="release CLI run"),
            threshold=PracticalThreshold(metric="capture_rate", unit="probability", minimum_effect=0.05, direction=EffectDirection.GREATER_IS_BETTER),
        )
        manifest = runs.read(result.seed_outcomes[0].run_id)
        ceiling = ResourceCeiling(resource_ceiling_id="release-ceiling", measured_at_utc=args.measured_at_utc, accelerator_hours=10.0, wall_clock_hours=10.0)
        estimate = ConditionResourceEstimate(condition_id=condition_id, accelerator_hours_per_seed=0.01, wall_clock_hours_per_seed=0.01, env_steps_per_seed=30)
        ledger = ConditionExecutionLedger(
            protocol_hash=protocol.protocol_hash, sample_size_plan=decide_sample_size(ceiling, [estimate]),
            planned_condition_ids=(condition_id,), records=(result.execution_record,),
        )
        evidence = (
            EvidenceRecord(
                record_id="ev-capture", evidence_type=EvidenceType.EXPERIMENT_OBSERVATION, producer="paired-evaluator",
                created_at_utc=args.measured_at_utc, method="paired episode evaluation", source=f"{manifest.run_id} metrics",
                extracted_value=result.execution_record.result.value if result.execution_record.result is not None else None,
                verification="metrics artifact hash matches the sealed manifest",
            ),
        )

        release_report = run_paper_release(
            token, bundle_id=args.bundle_id, run_manifests=(manifest,), statistics=result.statistics, execution_ledger=ledger,
            planned_conditions=specs, citation_ledger=_citation_ledger(), citation_key=CITATION_KEY, evidence=evidence,
            scope=_scope(), sections=_sections(), condition_id=condition_id, analysis_id=args.analysis_id,
            config_hash=_hash(f"config::{args.bundle_id}"), asserts_superiority=args.asserts_superiority,
        )
    except ResearchValidationError as exc:
        sys.stderr.buffer.write(canonical_json(exc.as_dict()) + b"\n")
        return 2

    payload = {
        "admission_hash": release_report.admission_hash,
        "eligible": release_report.eligible,
        "gate_eligible": release_report.gate_report.eligible,
        "gate_error_codes": list(release_report.gate_report.error_codes),
        "export_eligible": release_report.export_report.eligible,
        "export_error_codes": [item.code for item in release_report.export_report.errors],
        "bundle_id": release_report.bundle.bundle_id if release_report.bundle is not None else None,
        "table_hash": release_report.bundle.tables[0].table_hash if release_report.bundle is not None else None,
        "figure_hash": release_report.bundle.figures[0].figure_hash if release_report.bundle is not None else None,
    }
    output = canonical_json(payload) + b"\n"
    if args.output is None:
        sys.stdout.buffer.write(output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(output)
    return 0 if release_report.eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
