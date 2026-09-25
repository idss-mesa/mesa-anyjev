-- 0001_mesa_anyjev: the decision provenance sidecar (mesa-anyjev DESIGN D4).
--
-- Why a sidecar and not columns on mesa.avu_changes: the AVU triple is a hard contract with
-- iRODS; mesa_ducklake.AvuChange is extra='forbid' and its Parquet reader unpacks exactly 13
-- columns; a candidate set with a probability distribution, thresholds, human overrides and
-- labels does not fit a flat column. This schema never touches schema `mesa`. The join into
-- DuckLake is (project_id, snapshot_id, irods_path, attribute, value, unit); snapshot_id is
-- NULL for decisions and links that wrote nothing.
--
-- Postgres dialect. The DuckDB dialect lives in mesa_anyjev/provenance/store.py (JSON instead
-- of JSONB, TEXT instead of UUID/CHAR, no foreign keys); a test asserts both expose the same
-- column names and nullability.
BEGIN;

CREATE SCHEMA IF NOT EXISTS mesa_anyjev;

CREATE TABLE IF NOT EXISTS mesa_anyjev.schema_versions (
    version     INTEGER PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS mesa_anyjev.runs (
    run_id                UUID PRIMARY KEY,
    started_at            TIMESTAMPTZ NOT NULL,
    finished_at           TIMESTAMPTZ,
    status                TEXT NOT NULL CHECK (status IN ('decided','applied','partial','failed','dry_run')),
    actor                 TEXT NOT NULL,
    irods_path            TEXT,
    project_id            UUID,
    card_name             TEXT NOT NULL,
    card_sha256           TEXT NOT NULL,
    planner               TEXT NOT NULL,
    planner_model         TEXT,
    planner_prompt_sha256 TEXT,
    plan_json             JSONB NOT NULL,
    planner_fallback      BOOLEAN NOT NULL DEFAULT FALSE,
    backend_kind          TEXT NOT NULL,
    canonical_model       TEXT,
    served_model          TEXT,
    served_revision       TEXT,
    tokenizer             TEXT,
    tokenizer_revision    TEXT,
    anyjev_commit         TEXT NOT NULL,
    mesa_anyjev_version   TEXT NOT NULL,
    questions_lock_sha    TEXT NOT NULL,
    artifacts_sha256      TEXT,
    policy_profile        TEXT NOT NULL,
    config_sha256         TEXT NOT NULL,
    write_mode            TEXT NOT NULL DEFAULT 'none' CHECK (write_mode IN ('local','irods','none')),
    hosted_provider_used  BOOLEAN NOT NULL DEFAULT FALSE,
    data_left_host        BOOLEAN NOT NULL DEFAULT FALSE,
    egress_region         TEXT,
    missing_labels        INTEGER NOT NULL DEFAULT 0,
    n_prompts             INTEGER,
    seconds               REAL
);

CREATE TABLE IF NOT EXISTS mesa_anyjev.decisions (
    decision_id         UUID PRIMARY KEY,
    run_id              UUID NOT NULL REFERENCES mesa_anyjev.runs(run_id),
    seq                 INTEGER NOT NULL,
    group_id            UUID,
    parent_decision_id  UUID,
    question_id         TEXT NOT NULL,
    question_key        TEXT NOT NULL,
    kind                TEXT NOT NULL CHECK (kind IN ('choice','noul','score')),
    k                   INTEGER NOT NULL,
    scope               TEXT NOT NULL CHECK (scope IN ('dataset','column','site','avu')),
    column_name         TEXT,
    site_code           TEXT,
    state_sha256        TEXT NOT NULL,
    state_json          JSONB NOT NULL,
    prompt_sha256       TEXT,
    provider            TEXT NOT NULL,
    method              TEXT NOT NULL,
    model               TEXT NOT NULL,
    served_model        TEXT,
    level               TEXT NOT NULL CHECK (level IN ('raw','L0','L1','L2','none')),
    calibration         TEXT NOT NULL CHECK (calibration IN ('anyjev','typesafe','none')),
    probs               JSONB,                  -- NULL iff calibration = 'none'
    answer_index        INTEGER NOT NULL,       -- -1 = abstain
    answer              TEXT NOT NULL,
    confidence          REAL,
    p_true              REAL,                   -- noul only
    margin              REAL,
    score_value         REAL,                   -- score only
    answer_mass         REAL,
    prior_method        TEXT,
    order_flip_l0       REAL,
    permutations        INTEGER,
    temperature         REAL,
    n_calib             INTEGER,
    routed_from         TEXT,
    adapted             BOOLEAN,
    blocks_executed     INTEGER,
    masked_mass         REAL,
    threshold_auto      REAL,
    threshold_propose   REAL,
    outcome             TEXT NOT NULL CHECK (outcome IN ('auto','proposed','human','escalated','abstain','rejected','rule','decider_unavailable')),
    ts                  TIMESTAMPTZ NOT NULL,
    CHECK ((probs IS NULL) = (calibration = 'none'))
);
CREATE INDEX IF NOT EXISTS decisions_run_seq ON mesa_anyjev.decisions (run_id, seq);
CREATE INDEX IF NOT EXISTS decisions_key_ts ON mesa_anyjev.decisions (question_key, ts);
CREATE INDEX IF NOT EXISTS decisions_group ON mesa_anyjev.decisions (group_id);

CREATE TABLE IF NOT EXISTS mesa_anyjev.decision_options (
    decision_id   UUID NOT NULL REFERENCES mesa_anyjev.decisions(decision_id),
    option_index  INTEGER NOT NULL,
    option_text   TEXT NOT NULL,
    curie         TEXT,
    iri           TEXT,
    ontology_id   TEXT,
    prob          REAL,
    rank          INTEGER,
    PRIMARY KEY (decision_id, option_index)
);

CREATE TABLE IF NOT EXISTS mesa_anyjev.decision_groups (
    group_id            UUID PRIMARY KEY,
    run_id              UUID NOT NULL REFERENCES mesa_anyjev.runs(run_id),
    question_id         TEXT NOT NULL,
    scope               TEXT NOT NULL CHECK (scope IN ('dataset','column','site','avu')),
    column_name         TEXT,
    site_code           TEXT,
    aspect              TEXT,
    ontology_id         TEXT,
    search_json         JSONB NOT NULL,
    n_candidates        INTEGER NOT NULL,
    winner_decision_id  UUID,
    top_p               REAL,
    group_margin        REAL,
    level               TEXT,
    outcome             TEXT NOT NULL CHECK (outcome IN ('auto','proposed','human','escalated','abstain','rejected','rule','decider_unavailable')),
    missing_labels      INTEGER NOT NULL DEFAULT 0,
    escalated_from      UUID,
    ts                  TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS mesa_anyjev.avu_links (
    link_id       UUID PRIMARY KEY,
    run_id        UUID NOT NULL REFERENCES mesa_anyjev.runs(run_id),
    group_id      UUID,
    decision_id   UUID,
    project_id    UUID,
    snapshot_id   BIGINT,
    irods_path    TEXT,
    target_type   TEXT NOT NULL,
    attribute     TEXT NOT NULL,
    value         TEXT NOT NULL,
    unit          TEXT NOT NULL,
    op            TEXT NOT NULL DEFAULT 'add' CHECK (op IN ('add','delete')),
    term_curie    TEXT,
    term_iri      TEXT,
    term_label    TEXT,
    ontology_id   TEXT,
    aspect        TEXT,
    column_name   TEXT,
    site_code     TEXT,
    value_kind    TEXT,
    source        TEXT,
    write_status  TEXT NOT NULL CHECK (write_status IN ('proposed','rejected','accepted','written','mirror_failed','dry_run')),
    duplicate_of  UUID,
    written_at    TIMESTAMPTZ,
    UNIQUE NULLS NOT DISTINCT (run_id, attribute, value, unit, column_name, site_code)
);
CREATE INDEX IF NOT EXISTS avu_links_path ON mesa_anyjev.avu_links (irods_path, attribute, value, unit);
CREATE INDEX IF NOT EXISTS avu_links_snapshot ON mesa_anyjev.avu_links (project_id, snapshot_id);

CREATE TABLE IF NOT EXISTS mesa_anyjev.human_overrides (
    override_id     UUID PRIMARY KEY,
    group_id        UUID,
    decision_id     UUID,
    link_id         UUID,
    actor           TEXT NOT NULL,
    action          TEXT NOT NULL CHECK (action IN ('pick','accept','reject','decline','restore')),
    chosen_index    INTEGER,
    chosen_curie    TEXT,
    elicitation_key TEXT,
    offered         JSONB NOT NULL,
    ts              TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS mesa_anyjev.labels (
    label_id            UUID PRIMARY KEY,
    question_id         TEXT NOT NULL,
    question_key        TEXT NOT NULL,
    state_sha256        TEXT NOT NULL,
    state_json          JSONB NOT NULL,
    label_index         INTEGER NOT NULL,
    label_source        TEXT NOT NULL CHECK (label_source IN ('curator','curator_implicit','accepted_avu','consensus_all','consensus_majority','consensus_negative','teacher','hosted_jev','gold')),
    weight              REAL NOT NULL,
    source_ref          TEXT,
    card                TEXT,
    decision_id         UUID,
    batch_id            TEXT,
    consumed_in_bundle  TEXT,
    ts                  TIMESTAMPTZ NOT NULL,
    UNIQUE (question_key, state_sha256, label_source)
);
CREATE INDEX IF NOT EXISTS labels_key_source ON mesa_anyjev.labels (question_key, label_source);

INSERT INTO mesa_anyjev.schema_versions (version) VALUES (1) ON CONFLICT DO NOTHING;

COMMIT;
