"""Claude requests, recorded once and replayed forever.

A recording is filed under the sha256 of the request's canonical JSON: the
model, the settings, the system prompt, the passages and the question. Change
any of them and the key changes, so a replay can only ever return the answer to
the request actually being made. The replayer also compares the stored request
with the one asked for, byte for byte, before it trusts the file.

There is no live fallback. A missing recording raises ``RecordingMissing``, and
the page turns that into a refusal naming the command that records it.

Recording goes through the Message Batches API at half price, and batches do
not accept the server-side ``fallbacks`` parameter, so a declined batch request
is recorded as declined. ``live=True`` sends requests one at a time with
fallbacks on, and the recording names the model that answered.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..errors import DataSourceError, TechvalError

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def canonical(params: Mapping[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_key(params: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical(params).encode("utf-8")).hexdigest()


class RecordingMissing(TechvalError):
    def __init__(self, task_id: str, key: str) -> None:
        super().__init__(f"no recording for task {task_id} (request {key[:12]}); run `techval rag record`")
        self.task_id = task_id
        self.key = key


@dataclass(frozen=True)
class Recording:
    key: str
    task_id: str
    request: dict
    response: dict
    served_by: str | None
    usage: dict
    recorded_at: str
    mode: str
    sdk_version: str

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> Recording:
        missing = [f for f in cls.__dataclass_fields__ if f not in raw]
        if missing:
            raise DataSourceError(f"a recording is missing {', '.join(missing)}")
        return cls(**{f: raw[f] for f in cls.__dataclass_fields__})


class Replayer:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def recording(self, task_id: str, params: Mapping[str, Any]) -> Recording:
        key = request_key(params)
        path = self.path(key)
        if not path.exists():
            raise RecordingMissing(task_id, key)
        found = Recording.from_json(json.loads(path.read_text(encoding="utf-8")))
        if canonical(found.request) != canonical(params):
            raise DataSourceError(
                f"recording {key[:12]} holds a different request from the one task {task_id} sends"
            )
        return found


class Recorder:
    def __init__(
        self,
        client: Any,
        root: str | Path,
        *,
        today: str,
        sdk_version: str,
        sleep: Callable[[float], None] = time.sleep,
        poll_seconds: float = 30,
    ) -> None:
        self.client = client
        self.root = Path(root)
        self.today = today
        self.sdk_version = sdk_version
        self.sleep = sleep
        self.poll_seconds = poll_seconds
        self.failures: list[tuple[str, str]] = []

    def record(self, requests: Sequence[tuple[str, dict]], *, live: bool = False) -> list[Recording]:
        replayer = Replayer(self.root)
        pending = [(t, p) for t, p in requests if not replayer.path(request_key(p)).exists()]
        if not pending:
            return []
        self.root.mkdir(parents=True, exist_ok=True)
        return self._live(pending) if live else self._batch(pending)

    def _batch(self, pending: Sequence[tuple[str, dict]]) -> list[Recording]:
        by_key = {request_key(p): (t, p) for t, p in pending}
        batch = self.client.messages.batches.create(
            requests=[{"custom_id": key, "params": params} for key, (_, params) in by_key.items()]
        )
        while self.client.messages.batches.retrieve(batch.id).processing_status != "ended":
            self.sleep(self.poll_seconds)
        written = []
        for entry in self.client.messages.batches.results(batch.id):
            task_id, params = by_key[entry.custom_id]
            if entry.result.type != "succeeded":
                self.failures.append((task_id, entry.result.type))
                continue
            written.append(self._write(task_id, params, entry.result.message.to_dict(), "batch"))
        return sorted(written, key=lambda r: r.task_id)

    def _live(self, pending: Sequence[tuple[str, dict]]) -> list[Recording]:
        written = []
        for task_id, params in pending:
            message = self.client.beta.messages.create(
                **params, betas=[FALLBACK_BETA], extra_body={"fallbacks": "default"}
            )
            written.append(self._write(task_id, params, message.to_dict(), "live"))
        return written

    def _write(self, task_id: str, params: dict, message: dict, mode: str) -> Recording:
        recording = Recording(
            key=request_key(params),
            task_id=task_id,
            request=params,
            response=message,
            served_by=message.get("model"),
            usage=message.get("usage") or {},
            recorded_at=self.today,
            mode=mode,
            sdk_version=self.sdk_version,
        )
        Replayer(self.root).path(recording.key).write_text(
            json.dumps(recording.to_json(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return recording
