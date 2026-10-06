"""Tests for event messaging and distributed job dispatch."""

from cleave.events.producer import EventProducer, IngestionEvent


def test_event_producer_local_subscribers():
    producer = EventProducer(kafka_rest_url="http://127.0.0.1:9999")
    received = []

    def handler(event: IngestionEvent):
        received.append(event)

    producer.subscribe_local("doc_ingest", handler)

    event = IngestionEvent(
        job_id="job_123",
        file_uri="file:///test.pdf",
        filename="test.pdf",
        modality="document",
        options={"fast": True},
    )

    success = producer.publish("doc_ingest", event)
    assert success is True
    assert len(received) == 1
    assert received[0].job_id == "job_123"
    assert received[0].filename == "test.pdf"
