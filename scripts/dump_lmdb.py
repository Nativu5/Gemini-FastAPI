import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import lmdb
import orjson


def _decode_value(value: bytes) -> Any:
    """Decode a value from LMDB to Python data."""
    try:
        return orjson.loads(value)
    except orjson.JSONDecodeError:
        return value.decode("utf-8", errors="replace")


def _dump_all(txn: lmdb.Transaction) -> list[dict[str, Any]]:
    """Return all records from the database."""
    result: list[dict[str, Any]] = []
    for key, value in txn.cursor():
        result.append({"key": key.decode("utf-8"), "value": _decode_value(value)})
    return result


def _dump_selected(txn: lmdb.Transaction, keys: Iterable[str]) -> list[dict[str, Any]]:
    """Return records for the provided keys."""
    result: list[dict[str, Any]] = []
    for key in keys:
        raw = txn.get(key.encode("utf-8"))
        if raw is not None:
            result.append({"key": key, "value": _decode_value(raw)})
    return result


def dump_lmdb(path: Path, keys: Iterable[str] | None = None) -> None:
    """Print selected or all key-value pairs from the LMDB database."""
    env = lmdb.open(str(path), readonly=True, lock=False)
    with env.begin() as txn:
        records = _dump_selected(txn, keys) if keys else _dump_all(txn)
    env.close()

    print(orjson.dumps(records, option=orjson.OPT_INDENT_2).decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump LMDB records as JSON")
    parser.add_argument("path", type=Path, help="Path to LMDB directory")
    parser.add_argument("keys", nargs="*", help="Keys to retrieve")
    args = parser.parse_args()

    dump_lmdb(args.path, args.keys)


if __name__ == "__main__":
    main()
