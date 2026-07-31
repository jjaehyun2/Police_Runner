"""Research training APIs."""
from .checkpoint import (
    OPTIONAL_RESUME_FIELDS,
    REQUIRED_RESUME_FIELDS,
    RESUME_STATE_SCHEMA_VERSION,
    CompatibilityContract,
    PendingTransitionSnapshot,
    ResumeState,
    capture_trainer_resume_state,
    check_resume_compatibility,
    load_resume_state,
    restore_trainer_resume_state,
    resume_state_content_hash,
    save_resume_state,
)
from .equivalence import (
    EQUIVALENCE_SCHEMA_VERSION,
    EquivalenceReport,
    UpdateTrace,
    assert_equivalent,
    compare_traces,
    compare_traces_on_gpu_path,
    run_continuous,
    run_interrupted,
)
from .trainer import (
    DEFAULT_OUTPUT_ROOT,
    EpisodeRollout,
    PaperTrainer,
    ResearchTrainer,
    TRAINER_SCHEMA_VERSION,
    TrainerConfig,
    TrainingResult,
    UpdateOutcome,
    UpdateRecord,
)

__all__ = (
    "DEFAULT_OUTPUT_ROOT", "EQUIVALENCE_SCHEMA_VERSION", "EpisodeRollout", "EquivalenceReport",
    "OPTIONAL_RESUME_FIELDS", "PaperTrainer", "PendingTransitionSnapshot", "REQUIRED_RESUME_FIELDS",
    "RESUME_STATE_SCHEMA_VERSION", "ResearchTrainer", "ResumeState", "CompatibilityContract",
    "TRAINER_SCHEMA_VERSION", "TrainerConfig", "TrainingResult", "UpdateOutcome", "UpdateRecord",
    "UpdateTrace", "assert_equivalent", "capture_trainer_resume_state", "check_resume_compatibility",
    "compare_traces", "compare_traces_on_gpu_path", "load_resume_state", "restore_trainer_resume_state",
    "resume_state_content_hash", "run_continuous", "run_interrupted", "save_resume_state",
)
