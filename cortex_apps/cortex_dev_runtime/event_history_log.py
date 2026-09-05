"""
Event History Log (H_v): Chronological Provenance & Audit Trail.
===============================================================
Maintains an append-only, versioned log of code edits, test runs,
agent tool actions, and patch diffs.
Enables time-travel diffing, causal root-cause tracing, and
cross-version provenance inspection.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from cortex_apps.cortex_dev_runtime.dev_runtime_api import DevEvent, PatchDiff


class EventHistoryLog:
    """
    Chronological provenance store H_v.
    """

    def __init__(self):
        self.events: List[DevEvent] = []
        self.patches: Dict[str, PatchDiff] = {}
        self.version_to_event_idx: Dict[int, int] = {}
        self.entity_history: Dict[str, List[int]] = {}  # entity -> list of version numbers
        self.current_version: int = 0

    def append_event(
        self,
        event_type: str,
        target_path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> DevEvent:
        """Appends a new event and increments the substrate version."""
        self.current_version += 1
        ev = DevEvent(
            event_id=f"ev_{self.current_version:05d}",
            timestamp=time.time(),
            event_type=event_type,
            target_path=target_path,
            payload=payload or {},
            version=self.current_version,
        )
        self.events.append(ev)
        self.version_to_event_idx[self.current_version] = len(self.events) - 1
        self.entity_history.setdefault(target_path, []).append(self.current_version)
        return ev

    def record_patch(self, patch: PatchDiff) -> DevEvent:
        """Records a patch application event with its file diffs."""
        self.patches[patch.patch_id] = patch
        return self.append_event(
            event_type="PATCH_APPLIED",
            target_path=patch.patch_id,
            payload={
                "patch_id": patch.patch_id,
                "description": patch.description,
                "author": patch.author_agent,
                "files": list(patch.modified_files.keys()),
            },
        )

    def get_events_between(self, v1: int, v2: int) -> List[DevEvent]:
        """Returns events occurring strictly between version v1 and v2 (v1 < v <= v2)."""
        return [e for e in self.events if v1 < e.version <= v2]

    def get_history_for_entity(self, target_path: str) -> List[DevEvent]:
        """Returns all events modifying or referencing the given entity."""
        return [e for e in self.events if e.target_path == target_path or target_path in e.payload.get("files", [])]

    def get_latest_event(self) -> Optional[DevEvent]:
        return self.events[-1] if self.events else None

    def memory_bytes(self) -> int:
        """Estimates memory footprint of the event history log."""
        import sys
        base = sys.getsizeof(self.events) + sys.getsizeof(self.patches)
        for e in self.events:
            base += sys.getsizeof(e) + sys.getsizeof(e.payload)
        return base
