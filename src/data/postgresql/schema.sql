CREATE TABLE IF NOT EXISTS database_state (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    schema_version integer NOT NULL CHECK (schema_version > 0),
    last_ingest_seq bigint NOT NULL DEFAULT 0 CHECK (last_ingest_seq >= 0)
);

INSERT INTO database_state (schema_version) VALUES (1)
ON CONFLICT (singleton) DO NOTHING;

CREATE TABLE IF NOT EXISTS source_runs (
    source_run_id uuid PRIMARY KEY,
    model_sha256 text NOT NULL CHECK (model_sha256 ~ '^[0-9a-f]{64}$'),
    minizero_commit text NOT NULL CHECK (minizero_commit ~ '^([0-9a-f]{40}|[0-9a-f]{64})$'),
    effective_config text NOT NULL CHECK (btrim(effective_config) <> '')
);

CREATE TABLE IF NOT EXISTS games (
    game_id uuid PRIMARY KEY,
    source_run_id uuid NOT NULL REFERENCES source_runs (source_run_id),
    ingest_seq bigint NOT NULL UNIQUE CHECK (ingest_seq > 0),
    record_text text NOT NULL
);
