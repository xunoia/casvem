"""
Read pipeline — Step 1
Analyse query intent: extract topics, search terms, time relevance, memory type.
"""

from __future__ import annotations
import json
from providers.router import strong

_ANALYSE_PROMPT = """\
Analyse this user query and return ONLY valid JSON. No preamble, no explanation.

Required JSON format:
{{
  "topics": ["list of topic tags matching the query — use tags like: work, preferences, people, projects, finance, health, education, decisions, events"],
  "memory_type": "one of: preference | fact | project | event | person | decision | general",
  "time_relevance": "one of: current | historical | any",
  "search_terms": ["3–5 specific search phrases optimised for vector similarity search"]
}}

QUERY: {query}

JSON:"""


class QueryIntent:
    def __init__(self, topics, memory_type, time_relevance, search_terms):
        self.topics:         list[str] = topics
        self.memory_type:    str       = memory_type
        self.time_relevance: str       = time_relevance
        self.search_terms:   list[str] = search_terms


async def run(query: str) -> QueryIntent:
    prompt   = _ANALYSE_PROMPT.format(query=query)
    raw_json = await strong().generate_json(prompt)

    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, ValueError):
        data = {}

    return QueryIntent(
        topics         = data.get("topics", ["general"]),
        memory_type    = data.get("memory_type", "general"),
        time_relevance = data.get("time_relevance", "any"),
        search_terms   = data.get("search_terms", [query]),
    )
