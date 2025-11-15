"""Project package initialization."""

from pathlib import Path

from dotenv import load_dotenv


def _load_dotenv() -> None:
    """Load variables from the repository-level .env file, if present."""
    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)


_load_dotenv()
