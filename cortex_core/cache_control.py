"""Explicit cache-control boundary for inference backends.

The current runtime owns Hugging Face-style ``past_key_values`` objects in
Python. It can inspect, replace, and compact those objects, but it does not
own a scheduler or a native paged-attention block table. Backend-specific
implementations can conform to this boundary without changing compaction
policy code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Protocol, Tuple


@dataclass(frozen=True)
class CacheCapabilities:
    """Capabilities that callers may safely rely on for a cache backend."""

    backend: str
    native_paged: bool
    in_place_mutation: bool
    copy_on_write: bool
    topology_compaction: bool

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CachePolicy(str, Enum):
    """Comparison policies for cache-control experiments."""

    NONE = "none"
    TRUNCATE = "truncate"
    LANDMARK = "landmark"


@dataclass(frozen=True)
class CacheSnapshot:
    """Content-free cache telemetry suitable for logs and benchmarks."""

    backend: str
    sequence_length: int
    layer_count: int
    key_shapes: Tuple[Tuple[int, ...], ...]
    value_shapes: Tuple[Tuple[int, ...], ...]
    capabilities: CacheCapabilities

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["capabilities"] = self.capabilities.as_dict()
        return data


@dataclass(frozen=True)
class CachePolicyMeasurement:
    """Content-free observation from one cache policy trial."""

    policy: str
    elapsed_ms: float
    bytes_before: int
    bytes_after: int
    sequence_before: int
    sequence_after: int
    output_equivalent: Optional[bool]
    evaluation: Optional[Dict[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CacheControlAdapter(Protocol):
    """Operations required by cache-aware runtime components."""

    capabilities: CacheCapabilities

    def sequence_length(self, cache: Any) -> int:
        ...

    def snapshot(self, cache: Any) -> CacheSnapshot:
        ...

    def compact(self, cache: Any, synapse: Any, query_states: Any = None,
                keep_ratio: Optional[float] = None) -> Any:
        ...


class PythonKVCacheAdapter:
    """Adapter for tuple and ``DynamicCache`` objects managed in Python.

    This adapter is intentionally copy/replace oriented. A compaction returns
    a new landmark cache (or the original cache when no landmark result is
    available); it never claims to mutate backend-owned pages in place.
    """

    capabilities = CacheCapabilities(
        backend="huggingface-python-kv",
        native_paged=False,
        in_place_mutation=False,
        copy_on_write=False,
        topology_compaction=True,
    )

    @staticmethod
    def _layers(cache: Any) -> Iterable[Tuple[Any, Any]]:
        if cache is None:
            return ()
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            return zip(cache.key_cache, cache.value_cache)
        try:
            return tuple(layer for layer in cache if len(layer) >= 2)
        except (TypeError, IndexError):
            return ()

    def sequence_length(self, cache: Any) -> int:
        if cache is None:
            return 0
        if hasattr(cache, "get_seq_length"):
            try:
                return int(cache.get_seq_length())
            except (AttributeError, TypeError, ValueError):
                pass
        for key, _ in self._layers(cache):
            try:
                return int(key.shape[2])
            except (AttributeError, IndexError, TypeError):
                return 0
        return 0

    def cache_bytes(self, cache: Any) -> int:
        """Return tensor storage bytes visible through the adapter."""
        total = 0
        for key, value in self._layers(cache):
            for tensor in (key, value):
                try:
                    total += int(tensor.numel()) * int(tensor.element_size())
                except AttributeError:
                    continue
        return total

    def snapshot(self, cache: Any) -> CacheSnapshot:
        keys = []
        values = []
        for key, value in self._layers(cache):
            keys.append(tuple(int(size) for size in key.shape))
            values.append(tuple(int(size) for size in value.shape))
        return CacheSnapshot(
            backend=self.capabilities.backend,
            sequence_length=self.sequence_length(cache),
            layer_count=len(keys),
            key_shapes=tuple(keys),
            value_shapes=tuple(values),
            capabilities=self.capabilities,
        )

    def compact(self, cache: Any, synapse: Any, query_states: Any = None,
                keep_ratio: Optional[float] = None) -> Any:
        if cache is None:
            return None
        kwargs = {"query_states": query_states}
        if keep_ratio is not None:
            kwargs["keep_ratio"] = keep_ratio
        synapse.update_kv_landmarks(cache, **kwargs)
        compacted = synapse.get_landmarks()
        return compacted if compacted is not None else cache

    def truncate(self, cache: Any, keep_tokens: int) -> Any:
        """Return a tuple cache containing only the most recent tokens."""
        if keep_tokens < 1:
            raise ValueError("keep_tokens must be positive")
        layers = []
        for key, value in self._layers(cache):
            sequence_length = int(key.shape[2])
            start = max(sequence_length - keep_tokens, 0)
            layers.append((key[:, :, start:, :], value[:, :, start:, :]))
        return tuple(layers)

    def apply_policy(
        self,
        cache: Any,
        policy: CachePolicy | str,
        *,
        synapse: Any = None,
        query_states: Any = None,
        keep_tokens: int = 64,
        keep_ratio: Optional[float] = None,
    ) -> Any:
        """Apply one comparison policy without hiding backend limitations."""
        selected = CachePolicy(policy)
        if selected is CachePolicy.NONE:
            return cache
        if selected is CachePolicy.TRUNCATE:
            return self.truncate(cache, keep_tokens)
        if synapse is None:
            raise ValueError("landmark policy requires a synapse")
        return self.compact(
            cache, synapse, query_states=query_states, keep_ratio=keep_ratio,
        )

    def require_native_paged(self) -> None:
        """Fail loudly until a backend owns native paged-cache operations."""
        if not self.capabilities.native_paged:
            raise NotImplementedError(
                "native paged-KV control is not implemented by the Python cache adapter"
            )
