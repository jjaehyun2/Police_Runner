# Implementation Plan: OSM Road Pursuit Demo

## Overview

Python 기반 기존 도로 추격 코드를 확장해 OSM 준비·캐시·계약 검사·실험 실행·평가·내보내기·FastAPI v1을 점진적으로 연결한다. 각 구현 작업은 선행 산출물을 명시적으로 사용하며, 테스트 작업은 선택 항목(`*`)으로 표시한다. 레거시 체크포인트 직접 OSM 실행과 OSM 미세조정/재학습은 별도 실행 모드와 결과 저장소를 사용한다.

## Tasks

- [x] 1. Build domain foundations and evidence contracts
  - [x] 1.1 Pin and configure OSM demo dependencies and package layout
    - Add the `pursuit_evasion_rl/osm_demo` package and pin compatible FastAPI, Pydantic, HTTPX and OSMnx versions while retaining existing NetworkX, Gymnasium, PyTorch, matplotlib, pytest and Hypothesis integrations.
    - Add pytest markers for offline-default, live-OSM and experiment tests.
    - _Requirements: 2.2, 11.1, 12.1, 13.11, 13.12_

  - [x] 1.2 Implement versioned domain models, Daejeon preset and validators
    - Define immutable bbox, raw/model graph, metadata, mapping, checkpoint, episode, metrics, claim, run and export models.
    - Validate finite bbox ordering/area, schema versions, nonempty networks, exactly six police and valid positions.
    - Add the explicit bounded Daejeon drive-network preset and configuration version.
    - _Requirements: 2.1, 3.1–3.6, 9.1_

  - [x] 1.3 Implement canonical serialization, content hashes and cache-key construction
    - Create canonical JSON with finite-number checks, stable ordering and documented float quantization.
    - Compute SHA-256 artifact identities and cache keys without timestamps.
    - _Requirements: 2.3, 4.2, 5.8_

  - [x] 1.4 Implement experiment plans, run lineage and evidence-backed claim registry
    - Model immutable plans, region splits, legacy-direct/fine-tune/from-scratch/evaluation modes and claim transition guards.
    - Exclude legacy-direct, preliminary, failed and unreproducible artifacts from verified OSM performance evidence.
    - _Requirements: 1.1–1.5, 8.5–8.10, 10.7–10.8, 14.1–14.2, 14.5–14.6_

  - [ ]* 1.5 Write Property 5 test for deterministic cache keys
    - **Property 5: Cache keys are deterministic over all identity inputs**
    - **Validates: Requirements 4.2**

  - [ ]* 1.6 Write Property 7 test for supported input domains
    - **Property 7: Input validation accepts exactly the supported domain**
    - **Validates: Requirements 3.1–3.6, 13.7**

  - [ ]* 1.7 Write Property 16 test for evidence and claim boundaries
    - **Property 16: Evidence and claim states cannot overstate experiments**
    - **Validates: Requirements 1.1, 1.3–1.5, 8.5–8.6, 8.10, 10.7–10.8, 14.1, 14.5–14.6**

  - [ ]* 1.8 Write Property 17 test for training modes and region splits
    - **Property 17: Training modes and geographic splits remain distinct**
    - **Validates: Requirements 8.7–8.9, 14.2**

- [x] 2. Implement deterministic OSM preparation and offline bundles
  - [x] 2.1 Implement bounded OSM source and raw graph conversion
    - Fetch only explicit bboxes, clip oversized place results with `TRUNCATED`, project to metric CRS and retain directed edge geometry/tags/provenance.
    - Support injected fixture sources so default code paths require no external service.
    - _Requirements: 2.2–2.5, 13.12_

  - [x] 2.2 Implement decision-node coarsening, canonical IDs and mapping manifests
    - Detect branch, merge, dead-end, boundary and attribute-transition decisions; collapse compatible directed chains.
    - Generate raw-ID-independent canonical source signatures, contiguous IDs, stable segment/action ordering and full source mappings.
    - Reject ambiguous canonical collisions instead of using raw IDs as tiebreakers.
    - _Requirements: 5.1–5.8_

  - [x] 2.3 Implement degree splitting, topology validation and network statistics
    - Add deterministic acyclic virtual routing for out-degree above five without truncating exits.
    - Compare pre/post directed boundary reachability and calculate components, unreachable segments and validation reasons.
    - _Requirements: 5.9, 6.1–6.3, 6.6_

  - [x] 2.4 Implement immutable atomic cache bundles and verified loading
    - Stage raw graph, model graph, mapping and metadata; write a final bundle manifest and atomically publish immutable entries.
    - Validate schema and every hash on load, quarantine corrupt entries logically, preserve attribution and emit actionable offline misses.
    - _Requirements: 4.1–4.7_

  - [x] 2.5 Add offline raw/model/cache fixtures and migration boundaries
    - Create small one-way, bidirectional, parallel-edge, degree-six, boundary-crossing and Daejeon-shaped fixtures with committed hashes.
    - Detect existing single-file legacy cache data and require explicit migration rather than silent reuse.
    - _Requirements: 4.3–4.7, 13.10–13.12_

  - [ ]* 2.6 Write Property 1 test for canonicalization invariance
    - **Property 1: Canonicalization is deterministic and source-ID invariant**
    - **Validates: Requirements 5.6, 5.8, 13.3**

  - [ ]* 2.7 Write Property 2 test for directed provenance-preserving chains
    - **Property 2: Coarsened segments are valid directed provenance-preserving chains**
    - **Validates: Requirements 5.1–5.5, 5.7, 13.1**

  - [ ]* 2.8 Write Property 3 test for degree and reachability preservation
    - **Property 3: Degree handling preserves legal exits and boundary reachability**
    - **Validates: Requirements 5.9, 6.2–6.3**

  - [ ]* 2.9 Write Property 4 test for coarsening idempotence
    - **Property 4: Coarsening normalized output is idempotent**
    - **Validates: Requirements 13.4**

  - [ ]* 2.10 Write Property 6 test for cache round trips and mutation detection
    - **Property 6: Cache serialization round trips and detects mutation**
    - **Validates: Requirements 4.3, 4.6–4.7, 13.5**

  - [ ]* 2.11 Write Property 8 test for network statistics
    - **Property 8: Network statistics equal a reference graph model**
    - **Validates: Requirements 6.1, 6.6**

- [x] 3. Implement checkpoint contracts and topology-safe inference
  - [x] 3.1 Implement safe checkpoint inspection and compatibility reports
    - Compute checkpoint hashes, extract ordered `police_actor` tensor shapes and recognize the 21→128→128→6 structure.
    - Compare optional manifests while keeping police count, observation semantics, training region and performance `unknown` when not inferable.
    - Return sectioned machine-readable pass/warning/fail/unknown checks and mismatch details.
    - _Requirements: 7.1–7.7_

  - [x] 3.2 Implement separate legacy and OSM topology observation profiles
    - Reproduce `legacy_grid_v0` only for archaeology/explicit experiments.
    - Build `osm_topology_v1` as 21 finite normalized float32 topology/distance features with stable action slots and no raw/canonical ID feature values.
    - _Requirements: 7.2, 7.6, 7.8, 8.4, 8.11_

  - [x] 3.3 Implement action masks, learned policy loading and atomic recommendation decoding
    - Map five ordered exits plus stay, apply masks before probabilities and fail the whole response on masked/incomplete results.
    - Require six-police/profile/network/execution compatibility; isolate explicit legacy direct inference as experimental.
    - _Requirements: 6.4–6.5, 8.1–8.6_

  - [ ]* 3.4 Write Property 9 test for checkpoint introspection boundaries
    - **Property 9: Checkpoint inspection never infers unavailable semantics**
    - **Validates: Requirements 7.1–7.7**

  - [ ]* 3.5 Write Property 10 test for topology observation invariance
    - **Property 10: OSM observations are finite, normalized and relabel-invariant**
    - **Validates: Requirements 7.8, 8.4, 8.11, 13.6, 13.13**

  - [ ]* 3.6 Write Property 11 test for masks and complete recommendations
    - **Property 11: Action masks and recommendations correspond exactly to legal choices**
    - **Validates: Requirements 6.4–6.5, 8.1–8.3, 13.2**

- [x] 4. Implement OSM episodes, policies and deterministic runner
  - [x] 4.1 Implement an injected-network Gymnasium environment
    - Construct the environment from ModelNetwork, move vehicles along directed polyline arc length and resolve bounded zero-time virtual routing microsteps.
    - Enforce distinct seeded placement, metric capture, bbox-crossing escape, capture priority, timeout and logged stay behavior.
    - _Requirements: 3.3–3.5, 9.2–9.7, 9.9_

  - [x] 4.2 Adapt the heuristic fugitive to directed OSM networks
    - Use metric police visibility, directed distance to escape segments, deterministic action ordering and an injected random generator.
    - _Requirements: 9.1–9.3, 14.1_

  - [x] 4.3 Implement baseline and learned police policy interfaces
    - Add the deterministic shortest-path Baseline_Police and wrap compatible checkpoint inference behind the same team policy protocol.
    - Preserve legacy-direct and OSM-trained modes as different policy/run types.
    - _Requirements: 8.1–8.10, 14.3_

  - [x] 4.4 Implement single, batch and paired episode runners
    - Derive independent deterministic random streams, persist transition/event logs and replay identical placement/fugitive randomness across learned and baseline policies.
    - Return capture/escape/timeout records with input hashes and priority metadata.
    - _Requirements: 9.1–9.9, 14.2–14.4_

  - [ ]* 4.5 Write Property 12 test for placement and replay determinism
    - **Property 12: Placement and deterministic replay are reproducible**
    - **Validates: Requirements 9.1–9.2, 9.8**

  - [ ]* 4.6 Write Property 13 test for transitions and terminal precedence
    - **Property 13: Episode transitions respect directed movement and terminal precedence**
    - **Validates: Requirements 9.3–9.7, 9.9**

  - [ ]* 4.7 Write Property 18 test for paired policy comparisons
    - **Property 18: Paired policy comparisons use equivalent accounting**
    - **Validates: Requirements 14.3–14.4**

  - [ ]* 4.8 Add focused environment and policy example tests
    - Cover capture-and-escape same-step priority, boundary crossing, max-step timeout, virtual split traversal, dead-end stay, heuristic seeded randomness and baseline tie breaking.
    - _Requirements: 9.4–9.9, 13.9_

- [x] 5. Implement metrics, rendering and competition artifacts
  - [x] 5.1 Implement outcome, confidence and latency metrics
    - Enforce one outcome and count conservation; calculate rates, configured confidence intervals and episode distributions.
    - Measure six-action inference with `perf_counter_ns`, explicit warmups and device metadata; store `-1/not_measured` outside policy inference.
    - _Requirements: 10.1–10.9_

  - [x] 5.2 Implement OSM matplotlib frame and summary rendering
    - Draw projected directed roads, six police, fugitive, current segments, trails and a scale-correct capture circle.
    - Add terminal annotations and deterministic Korean-font discovery with English fallback.
    - _Requirements: 11.1–11.4, 11.6, 11.8_

  - [x] 5.3 Implement staged competition export generation
    - Export ordered start-to-terminal PNG frames, outcome/length/latency/path summaries and a content-addressed manifest.
    - Preserve OSM attribution, input hashes, code version, claim statuses, limitations, hypotheses, training path and research-only disclaimer.
    - _Requirements: 4.7, 11.5–11.8, 14.7–14.8_

  - [ ]* 5.4 Write Property 14 test for outcome conservation
    - **Property 14: Outcome aggregation conserves episode membership**
    - **Validates: Requirements 10.1–10.4, 13.8**

  - [ ]* 5.5 Write Property 15 test for latency summaries
    - **Property 15: Latency summaries match reference statistics**
    - **Validates: Requirements 10.5–10.6, 10.9**

  - [ ]* 5.6 Write Property 19 test for complete export manifests
    - **Property 19: Export manifests are complete and content-addressed**
    - **Validates: Requirements 11.7, 14.7–14.8**

  - [ ]* 5.7 Add renderer and export example/integration tests
    - Inspect artist data, equal metric aspect, capture radius, terminal labels, font fallback, ordered PNG names, figure set and artifact hashes on offline fixtures.
    - _Requirements: 11.1–11.8_

- [x] 6. Expose application services through FastAPI v1
  - [x] 6.1 Implement dependency-injected network, run and export services
    - Replace direct module-global orchestration with repository interfaces and process-local/file-backed demo implementations.
    - Wire online/offline preparation, compatibility, recommendation, run, metrics and export use cases.
    - _Requirements: 12.1–12.3_

  - [x] 6.2 Implement Pydantic v1 request/response contracts and routes
    - Add `/api/v1` health, network OSM/cache load, network info, compatibility, recommendation, run, metrics and export endpoints.
    - Return accepted run IDs, immutable hashes, seeds and result URLs; return six complete recommendations and measured latency.
    - _Requirements: 12.1–12.3_

  - [x] 6.3 Implement API error mapping, sanitization and generated examples
    - Map field validation to 422, missing IDs to 404, incompatibility to 409, dependency absence to 503 and sanitized unexpected errors to 500 correlation IDs.
    - Add request/success/error examples and schema descriptions to OpenAPI without absolute local paths.
    - _Requirements: 12.4–12.8_

  - [ ]* 6.4 Add FastAPI contract and error tests
    - Cover valid requests, bbox limits, bad canonical IDs, police-count mismatch, compatibility failure, offline cache, unified 404, 422 field paths, 409 guidance and sanitized 500.
    - _Requirements: 12.2–12.9_

  - [ ]* 6.5 Add the offline end-to-end integration test
    - Exercise fixture load, coarsening, cache verification, compatibility, episode, metrics, rendering, API and export with external OSM access forced to fail.
    - _Requirements: 13.10, 13.12_

- [x] 7. Add OSM training/evaluation entry points and reproducibility documentation
  - [x] 7.1 Implement manifest-driven OSM fine-tune and from-scratch entry points
    - Reuse the MAPPO architecture with six police, observation 21, action 6 and `[128,128]`; require `osm_topology_v1` and disjoint region manifests.
    - Save distinct lineage, normalization/action-order contracts and checkpoint manifests for fine-tune versus from-scratch runs.
    - _Requirements: 8.7–8.10, 14.2_

  - [x] 7.2 Implement pre-registered paired evaluation entry point
    - Freeze experiment config before execution, evaluate seen/unseen areas separately and compare learned and baseline policies on paired seeds.
    - Emit run/metrics/claim inputs without automatically marking successful criteria as verified.
    - _Requirements: 10.7–10.8, 14.1–14.6_

  - [x] 7.3 Implement reproducibility and API/export documentation artifacts
    - Add checked configuration examples for Daejeon preparation, offline cache, legacy-direct experimental mode, OSM training, paired evaluation and competition export.
    - Generate API examples from Pydantic/OpenAPI contracts and a competition report template separating implemented, verified, experimental, unsupported and limitations sections.
    - _Requirements: 1.1–1.5, 4.5, 8.5–8.10, 12.8, 14.7–14.8_

  - [ ]* 7.4 Add training/evaluation manifest integration tests
    - Verify split overlap rejection, lineage separation, missing-test experimental status, plan immutability, paired seed reuse and unreproducible evidence exclusion using tiny offline networks and mock training.
    - _Requirements: 8.7–8.10, 14.1–14.6_

  - [ ]* 7.5 Add opt-in bounded live OSM provenance tests
    - Query no more than three small configured bboxes and verify clipping status, source metadata and cache preparation without asserting policy performance.
    - _Requirements: 2.2–2.4, 13.11_

- [x] 8. Integration checkpoint - Ensure all automated tests pass
  - Run non-watch pytest/Hypothesis, type/lint checks available in the repository and FastAPI schema generation; keep live OSM and real training experiments opt-in.
  - Ensure all tests pass, ask the user if questions arise.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["1.3", "2.1"] },
    { "id": 3, "tasks": ["1.4", "2.2", "3.1"] },
    { "id": 4, "tasks": ["2.3"] },
    { "id": 5, "tasks": ["2.4", "3.2", "4.1"] },
    { "id": 6, "tasks": ["2.5", "3.3", "4.2"] },
    { "id": 7, "tasks": ["4.3"] },
    { "id": 8, "tasks": ["4.4"] },
    { "id": 9, "tasks": ["5.1", "7.1"] },
    { "id": 10, "tasks": ["5.2", "7.2"] },
    { "id": 11, "tasks": ["5.3"] },
    { "id": 12, "tasks": ["6.1"] },
    { "id": 13, "tasks": ["6.2"] },
    { "id": 14, "tasks": ["6.3"] },
    { "id": 15, "tasks": ["7.3"] },
    { "id": 16, "tasks": ["8"] }
  ]
}
```

## Notes

- Tasks marked with `*` are optional automated test tasks and can be skipped for a faster prototype; core implementation tasks are not optional.
- Every design correctness property has one dedicated Hypothesis task configured for at least 100 examples and tagged with the required feature/property comment.
- The DAG lists executable leaf tasks and the final checkpoint. Parent headings group work but are not dependency nodes.
- Live OSM retrieval, real model training and performance experiments remain opt-in; this plan creates the code and reproducibility artifacts but does not claim their results in advance.
