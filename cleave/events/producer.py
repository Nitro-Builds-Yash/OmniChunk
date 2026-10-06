"""Event Producer: Dispatches ingestion tasks to Kafka/Redpanda topic or in-memory queues."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from ..http import client

log = logging.getLogger(__name__)


@dataclass
class IngestionEvent:
    job_id: str
    file_uri: str
    filename: str
    modality: str
    options: dict[str, Any]


class EventProducer:
    """Publishes ingestion events to Kafka/Redpanda REST proxy or local queue."""

    def __init__(self, kafka_rest_url: str | None = None) -> None:
        self.kafka_rest_url = kafka_rest_url or os.environ.get(
            "REDPANDA_REST_URL", "http://127.0.0.1:9644"
        )
        self._local_subscribers: dict[str, list[Callable[[IngestionEvent], None]]] = {}
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            r = client().get(f"{self.kafka_rest_url}/v1/status/ready", timeout=1.0)
            self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def subscribe_local(self, topic: str, handler: Callable[[IngestionEvent], None]) -> None:
        """Register an in-memory handler for offline / test operation."""
        self._local_subscribers.setdefault(topic, []).append(handler)

    def publish(self, topic: str, event: IngestionEvent) -> bool:
        """Publish event to topic. Falls back to local in-memory listeners."""
        payload = asdict(event)

        # 1. Attempt Redpanda/Kafka REST proxy if available
        if self.is_available():
            try:
                r = client().post(
                    f"{self.kafka_rest_url}/topics/{topic}",
                    json={"records": [{"value": payload}]},
                    headers={"Content-Type": "application/vnd.kafka.json.v2+json"},
                    timeout=5.0,
                )
                if r.status_code in (200, 201):
                    log.info("Dispatched event to %s: %s", topic, event.filename)
                    return True
            except Exception as exc:
                log.debug("Kafka publish failed (%s); using in-memory subscribers", exc)

        # 2. In-memory dispatch
        handlers = self._local_subscribers.get(topic, [])
        for h in handlers:
            try:
                h(event)
            except Exception as exc:
                log.error("Local event handler error on topic %s: %s", topic, exc)
        return True


_default_producer: EventProducer | None = None


def get_event_producer() -> EventProducer:
    global _default_producer
    if _default_producer is None:
        _default_producer = EventProducer()
    return _default_producer
