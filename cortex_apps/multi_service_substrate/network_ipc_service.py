"""
Network IPC & Socket Serialization Service.
===========================================
Implements real TCP loopback socket communication and binary/JSON wire serialization.
Measures actual wire bytes transferred, operating system socket syscalls,
and round-trip IPC transit time.
"""

from __future__ import annotations

import json
import select
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from cortex_apps.multi_service_substrate.fragmented_production import FragmentedProductionArchitecture
from cortex_apps.multi_service_substrate.substrate_api import (
    ContextPack,
    ContextSubstrate,
    EntityStatus,
    OperationMetrics,
    ProposedAction,
    SubstrateSnapshot,
    TelemetryEvent,
    VerificationResult,
)
from cortex_apps.research_agent_system.world_state import ResearchDocument, ResearchWorldCatalog


class NetworkIPCServer:
    """Background TCP socket server hosting a remote fragmented context worker."""

    def __init__(self, target_substrate: ContextSubstrate, host: str = "127.0.0.1", port: int = 0):
        self.target_substrate = target_substrate
        self.host = host
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, port))
        self.port = self.server_sock.getsockname()[1]
        self.server_sock.listen(128)
        self.running = True
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while self.running:
            try:
                r, _, _ = select.select([self.server_sock], [], [], 0.2)
                if not r:
                    continue
                conn, _ = self.server_sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                t = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
                t.start()
            except Exception:
                break

    def _handle_client(self, conn: socket.socket):
        try:
            while self.running:
                raw_len = conn.recv(4)
                if not raw_len or len(raw_len) < 4:
                    break
                msg_len = int.from_bytes(raw_len, "big")
                chunks = []
                bytes_recd = 0
                while bytes_recd < msg_len:
                    chunk = conn.recv(min(msg_len - bytes_recd, 4096))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    bytes_recd += len(chunk)

                req_bytes = b"".join(chunks)
                req = json.loads(req_bytes.decode("utf-8"))
                cmd = req.get("cmd")

                resp: Dict[str, Any] = {"status": "OK"}
                if cmd == "ingest":
                    ev_dict = req["event"]
                    event = TelemetryEvent(
                        event_id=ev_dict["event_id"],
                        timestamp=ev_dict["timestamp"],
                        event_type=ev_dict["event_type"],
                        entity_id=ev_dict["entity_id"],
                        raw_text=ev_dict["raw_text"],
                        metadata=ev_dict.get("metadata", {}),
                    )
                    v = self.target_substrate.ingest(event)
                    resp["version"] = v
                elif cmd == "verify":
                    act_dict = req["action"]
                    action = ProposedAction(
                        action_id=act_dict["action_id"],
                        action_name=act_dict["action_name"],
                        target_node=act_dict["target_node"],
                        required_prerequisites=act_dict["required_prerequisites"],
                    )
                    ver = self.target_substrate.verify(action)
                    resp["permit"] = ver.permit
                    resp["reason"] = ver.reason
                    resp["version"] = ver.version

                resp_bytes = json.dumps(resp).encode("utf-8")
                conn.sendall(len(resp_bytes).to_bytes(4, "big") + resp_bytes)
        except Exception:
            pass
        finally:
            conn.close()

    def shutdown(self):
        self.running = False
        try:
            self.server_sock.close()
        except Exception:
            pass


class FragmentedNetworkArchitecture(ContextSubstrate):
    """
    Contender A: Fragmented Services communicating through real TCP loopback sockets.
    Wraps worker execution with actual socket network transit, OS syscalls,
    and wire serialization.
    """

    def __init__(self, catalog: ResearchWorldCatalog):
        self.catalog = catalog
        self.underlying = FragmentedProductionArchitecture(catalog, sync_barrier=True)
        self.server = NetworkIPCServer(self.underlying)
        self.port = self.server.port

        self.actual_wire_bytes = 0
        self.socket_syscall_count = 0
        self.ipc_latency_ms = 0.0

        # Persistent client socket
        self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.client_sock.connect(("127.0.0.1", self.port))

    def _rpc_call(self, cmd: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        req_bytes = json.dumps({"cmd": cmd, **payload}).encode("utf-8")
        wire_msg = len(req_bytes).to_bytes(4, "big") + req_bytes

        self.client_sock.sendall(wire_msg)
        self.socket_syscall_count += 1
        self.actual_wire_bytes += len(wire_msg)

        raw_len = self.client_sock.recv(4)
        self.socket_syscall_count += 1
        self.actual_wire_bytes += 4

        msg_len = int.from_bytes(raw_len, "big")
        resp_bytes = self.client_sock.recv(msg_len)
        self.socket_syscall_count += 1
        self.actual_wire_bytes += len(resp_bytes)

        elapsed = (time.perf_counter() - t0) * 1000.0
        self.ipc_latency_ms += elapsed

        return json.loads(resp_bytes.decode("utf-8"))

    def ingest(self, event: TelemetryEvent) -> int:
        ev_dict = {
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "entity_id": event.entity_id,
            "raw_text": event.raw_text,
            "metadata": event.metadata,
        }
        res = self._rpc_call("ingest", {"event": ev_dict})
        return res["version"]

    def context(self, query: str, token_budget: int, version: Optional[int] = None) -> ContextPack:
        # Context uses local cache projection
        return self.underlying.context(query, token_budget, version)

    def route(self, event: TelemetryEvent, version: Optional[int] = None) -> Tuple[List[str], int]:
        return self.underlying.route(event, version)

    def affected(self, entity_id: str, version: Optional[int] = None) -> Tuple[List[str], int]:
        return self.underlying.affected(entity_id, version)

    def search(self, query: str, top_k: int = 5, version: Optional[int] = None) -> Tuple[List[ResearchDocument], int]:
        return self.underlying.search(query, top_k, version)

    def verify(self, action: ProposedAction, version: Optional[int] = None) -> VerificationResult:
        act_dict = {
            "action_id": action.action_id,
            "action_name": action.action_name,
            "target_node": action.target_node,
            "required_prerequisites": action.required_prerequisites,
        }
        res = self._rpc_call("verify", {"action": act_dict})
        return VerificationResult(
            permit=res["permit"],
            reason=res["reason"],
            version=res["version"],
        )

    def subscribe(self, predicate: Callable[[TelemetryEvent], bool], callback: Callable[[TelemetryEvent, int], None]) -> str:
        return self.underlying.subscribe(predicate, callback)

    def get_snapshot(self, version: Optional[int] = None) -> SubstrateSnapshot:
        return self.underlying.get_snapshot(version)

    def reset_metrics(self) -> None:
        self.actual_wire_bytes = 0
        self.socket_syscall_count = 0
        self.ipc_latency_ms = 0.0
        self.underlying.reset_metrics()

    def get_metrics(self) -> OperationMetrics:
        m = self.underlying.get_metrics()
        m.cpu_time_ms += self.ipc_latency_ms
        return m

    def shutdown(self):
        try:
            self.client_sock.close()
        except Exception:
            pass
        self.server.shutdown()
