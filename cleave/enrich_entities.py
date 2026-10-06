"""Entity and Relationship Enrichment Layer:
Extracts named entities, topics, and cross-chunk relationships using Gemini,
enriching the KnowledgeUnits and persisting connections to the Graph DB.
"""

from __future__ import annotations

import json
import logging

from .llm import get_provider
from .models import KnowledgeUnit

log = logging.getLogger(__name__)

NER_PROMPT_TEMPLATE = """Extract the key named entities (organizations, people, products, technical concepts, datasets) from the following chunk.
Also identify if this chunk explicitly references any other concept or section.

Return JSON in this format:
{
  "entities": ["entity1", "entity2", ...],
  "topics": ["topic1", ...],
  "summary": "Brief 1-sentence synopsis"
}

Text:
{text}
"""


def enrich_entities_batch(units: list[KnowledgeUnit], max_enrich: int = 10) -> list[KnowledgeUnit]:
    """Extract entities for key knowledge units using the configured LLM provider."""
    provider = get_provider()
    if not provider.is_configured() or provider.name == "none":
        log.debug("LLM provider not active; skipping entity enrichment")
        return units

    count = 0
    for u in units:
        if count >= max_enrich:
            break
        # Skip small code blocks or trivial chunks
        if u.token_count < 30 or u.content.startswith("# Schema"):
            continue

        prompt = NER_PROMPT_TEMPLATE.replace("{text}", u.content[:2000])
        try:
            resp_text, _usage = provider.complete_json(prompt)
            if resp_text:
                data = json.loads(resp_text)
                extracted_entities = data.get("entities", [])
                if extracted_entities:
                    # Merge unique entities
                    existing = set(u.entities)
                    for ent in extracted_entities:
                        if ent not in existing:
                            u.entities.append(ent)
                    u.metadata.setdefault("topics", data.get("topics", []))
                    count += 1
        except Exception as exc:
            log.debug("Entity enrichment failed for unit %s: %s", u.id, exc)

    log.info("Enriched entities for %d units via %s", count, provider.name)
    return units
