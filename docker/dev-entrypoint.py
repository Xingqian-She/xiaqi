"""Run the local PostgreSQL cluster for the lifetime of the dev command."""

import fcntl
import os
import pwd
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


POSTGRES_MAJOR = os.environ["PG_MAJOR"]
POSTGRES_ROOT = Path("/var/lib/postgresql")
POSTGRES_DATA = POSTGRES_ROOT / POSTGRES_MAJOR / "main"
CLUSTER_COMMAND = ["pg_ctlcluster", "--skip-systemctl-redirect", POSTGRES_MAJOR, "main"]


def prepare_database() -> None:
    postgres = pwd.getpwnam("postgres")
    if (POSTGRES_DATA / "PG_VERSION").is_file():
        # The UID assigned by the OS package can differ between images.
        data_owner = POSTGRES_DATA.stat()
        if (data_owner.st_uid, data_owner.st_gid) != (postgres.pw_uid, postgres.pw_gid):
            subprocess.run(
                ["chown", "--recursive", "--no-dereference", "postgres:postgres", POSTGRES_DATA],
                check=True,
            )
        return

    POSTGRES_DATA.parent.mkdir(parents=True, exist_ok=True)
    # Publish the data directory only after initdb succeeds.
    with tempfile.TemporaryDirectory(prefix=".initdb-", dir=POSTGRES_DATA.parent) as directory:
        os.chown(directory, postgres.pw_uid, postgres.pw_gid)
        subprocess.run(
            [
                "runuser", "-u", "postgres", "--",
                f"/usr/lib/postgresql/{POSTGRES_MAJOR}/bin/initdb",
                "--pgdata", directory,
                "--locale=C.UTF-8",
                "--encoding=UTF8",
                "--auth-local=peer",
                "--auth-host=scram-sha-256",
                "--no-instructions",
            ],
            check=True,
        )
        Path(directory).rename(POSTGRES_DATA)


def main() -> int:
    dev_command = None
    stop_signal = None

    def forward_signal(signum, _frame):
        nonlocal stop_signal
        if dev_command is None:
            # Let PostgreSQL finish starting before requesting its shutdown.
            stop_signal = signum
            return
        dev_command.send_signal(signum)
        if signum == signal.SIGTERM:
            raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)

    # The server PID file alone cannot coordinate separate container PID namespaces.
    with (POSTGRES_ROOT / ".xiaqi-postgresql.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("The PostgreSQL data volume is in use. Close the other dev container or open a shell in it.")

        prepare_database()
        if stop_signal is not None:
            return 128 + stop_signal
        subprocess.run([*CLUSTER_COMMAND, "start"], check=True)
        try:
            if stop_signal is not None:
                return 128 + stop_signal

            dev_command = subprocess.Popen(sys.argv[1:])
            if stop_signal is not None:
                dev_command.send_signal(stop_signal)
                return 128 + stop_signal

            returncode = dev_command.wait()
            return returncode if returncode >= 0 else 128 - returncode
        finally:
            dev_command = None
            subprocess.run([*CLUSTER_COMMAND, "stop"], check=False)


if __name__ == "__main__":
    sys.exit(main())
