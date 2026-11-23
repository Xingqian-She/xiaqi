#!/usr/bin/env bash
set -euo pipefail

usage='Usage: bash scripts/train.sh CONFIG [--resume CHECKPOINT]'
case "${1:-}" in
    -h|--help)
        printf '%s\n' "$usage"
        exit 0
        ;;
    '')
        printf '%s\n' "$usage" >&2
        exit 2
        ;;
esac

config_path="$1"
shift
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -f "$config_path" ]]; then
    printf 'Training config not found: %s\n' "$config_path" >&2
    exit 2
fi

if [[ -z "${XIAQI_DATABASE_URL:-}" ]]; then
    socket_dir=/var/run/postgresql
    database_user=$(id -un)
    runuser -u postgres -- psql -X -q -h "$socket_dir" -U postgres -d postgres \
        -v ON_ERROR_STOP=1 -v "database_user=$database_user" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN', :'database_user')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'database_user')
\gexec

SELECT format('CREATE DATABASE xiaqi OWNER %I', :'database_user')
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'xiaqi')
\gexec
SQL

    export XIAQI_DATABASE_URL="dbname=xiaqi host=$socket_dir user=$database_user"
    psql -X -q "$XIAQI_DATABASE_URL" -v ON_ERROR_STOP=1 -1 -f src/data/postgresql_store/schema.sql
fi

export PYTHONPATH="$PWD/build/minizero/go${PYTHONPATH:+:$PYTHONPATH}"
exec python -m scripts.train --config "$config_path" "$@"
