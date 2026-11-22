CREATE TABLE IF NOT EXISTS pools (
    pool_id text PRIMARY KEY CHECK (btrim(pool_id) <> ''),
    last_ingest_seq bigint NOT NULL DEFAULT 0 CHECK (last_ingest_seq >= 0)
);

CREATE TABLE IF NOT EXISTS source_runs (
    source_run_id uuid PRIMARY KEY,
    pool_id text NOT NULL REFERENCES pools (pool_id),
    model_sha256 text NOT NULL CHECK (model_sha256 ~ '^[0-9a-f]{64}$'),
    minizero_commit text NOT NULL CHECK (minizero_commit ~ '^([0-9a-f]{40}|[0-9a-f]{64})$'),
    effective_config text NOT NULL CHECK (btrim(effective_config) <> ''),
    UNIQUE (pool_id, source_run_id)
);

CREATE TABLE IF NOT EXISTS games (
    game_id uuid PRIMARY KEY,
    pool_id text NOT NULL,
    source_run_id uuid NOT NULL,
    ingest_seq bigint NOT NULL CHECK (ingest_seq > 0),
    record_text text NOT NULL,
    FOREIGN KEY (pool_id, source_run_id) REFERENCES source_runs (pool_id, source_run_id),
    UNIQUE (pool_id, ingest_seq)
);
