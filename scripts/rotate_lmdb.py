import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Run as a plain script as well as `python -m scripts.rotate_lmdb`: only the latter puts the
# repository root on the path, and `app` has to be importable either way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importing the store pulls in `app.utils`, which builds the application config at import time
# and exits when it cannot. This command touches nothing the Gemini section configures, so when
# there is no config file to read it seeds the one required field rather than refusing to run -
# rotating a detached or backup database must not require the server's configuration. A real
# config file still takes effect, because this only fills in what is otherwise missing.
if not Path(os.getenv("CONFIG_PATH", "config/config.yaml")).is_file():
    os.environ.setdefault("CONFIG_GEMINI", '{"clients": []}')

from app.services.lmdb import LMDBConversationStore


def _parse_duration(value: str) -> timedelta:
    """Parse duration in the format '14d' or '24h'."""
    if value.endswith("d"):
        return timedelta(days=int(value[:-1]))
    if value.endswith("h"):
        return timedelta(hours=int(value[:-1]))
    raise ValueError("Invalid duration format. Use Nd or Nh")


DEFAULT_MAP_SIZE = 1024 * 1024 * 1024


def rotate_lmdb(path: Path, keep: str, map_size: int = DEFAULT_MAP_SIZE) -> int:
    """Delete conversations last updated before the retention window, or all of them.

    Returns the number removed. `keep` is a duration like `14d`/`24h`, or `all` to empty
    the store. `map_size` is passed explicitly rather than read from the application config,
    so this command can rotate a detached or backup database whose size has nothing to do
    with the running server's settings.
    """
    # Opening an absent path would create an empty database and report a successful rotation of
    # nothing, so a mistyped path has to fail instead.
    if not (path / "data.mdb").is_file():
        raise SystemExit(f"No LMDB database at {path}")

    store = LMDBConversationStore.open_isolated(
        db_path=str(path), max_db_size=map_size, retention_days=0
    )
    try:
        if keep == "all":
            return store.clear()

        delta = _parse_duration(keep)
        threshold = datetime.now() - delta
        return store.cleanup_before(threshold)
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove outdated LMDB records")
    parser.add_argument("path", type=Path, help="Path to LMDB directory")
    parser.add_argument(
        "keep",
        help="Retention period, e.g. 14d or 24h. Use 'all' to delete every record",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=DEFAULT_MAP_SIZE,
        help=(
            f"LMDB map size in bytes for opening the target database (default: {DEFAULT_MAP_SIZE})"
        ),
    )
    args = parser.parse_args()

    removed = rotate_lmdb(args.path, args.keep, args.map_size)
    print(f"Removed {removed} conversation(s) from {args.path}")


if __name__ == "__main__":
    main()
