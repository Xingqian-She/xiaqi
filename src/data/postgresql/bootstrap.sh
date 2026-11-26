#!/usr/bin/env bash
set -euo pipefail

mode="$1"
shift
socket_dir=/var/run/postgresql
database_user=$(id -un)
export XIAQI_DATABASE_URL="dbname=xiaqi host=$socket_dir user=$database_user"
query=(psql -X -q -v ON_ERROR_STOP=1 "$XIAQI_DATABASE_URL")

if [[ "$mode" == init ]]; then
    runuser -u postgres -- psql -X -q -h "$socket_dir" -U postgres -d postgres \
        -v ON_ERROR_STOP=1 -v "database_user=$database_user" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN', :'database_user')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'database_user')
\gexec

SELECT format('CREATE DATABASE xiaqi OWNER %I', :'database_user')
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'xiaqi')
\gexec
SQL
    empty=$("${query[@]}" -Atc "SELECT NOT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public')")
    if [[ "$empty" == t ]]; then
        "${query[@]}" -1 -f "$(dirname "${BASH_SOURCE[0]}")/schema.sql"
    fi
fi

version=$("${query[@]}" -Atc 'SELECT schema_version FROM database_state WHERE singleton')
if [[ "$version" != 1 ]]; then
    printf 'Expected experiment database schema version 1, found %s.\n' "$version" >&2
    exit 1
fi

exec "$@"
