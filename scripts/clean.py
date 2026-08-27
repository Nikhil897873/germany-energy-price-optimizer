"""Remove only generated GridShift runtime artifacts."""

from pathlib import Path

TARGETS = (
    Path("data/bronze"),
    Path("data/silver"),
    Path("data/artifacts"),
    Path("dbt/logs"),
    Path("dbt/target"),
)


def main() -> None:
    for directory in TARGETS:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*"), reverse=True):
            if path.is_file() and path.name != ".gitkeep":
                path.unlink()
            elif path.is_dir() and not any(path.iterdir()):
                path.rmdir()
    for suffix in ("gridshift.duckdb", "gridshift.duckdb.wal"):
        path = Path("data") / suffix
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    main()
