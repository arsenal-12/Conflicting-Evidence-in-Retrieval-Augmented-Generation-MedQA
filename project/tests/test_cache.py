from types import SimpleNamespace
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evidence_cache import EvidenceCache

cache = EvidenceCache(
    cache_dir="experiment_cache/evidence",   # <-- use YOUR actual cache path
    generator=None,
    verifier=None,
    strict_cache=True,
)

print(cache.summary())