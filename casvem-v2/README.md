# CaSVeM v2 — Second Iteration (Archived)

Improved over v1 but still not the final architecture.

**Key improvements over v1**:
- Removed graph layer, pure vector retrieval
- Multi-provider LLM abstraction (OpenAI + Gemini)
- Better benchmark tooling

**Why archived**: Cache was still stateless — no learning from query patterns.
The breakthrough insight (cache warms over time, cost approaches zero) came in v3.

**Active version**: See [casvem-v3](../casvem-v3/) for the production architecture.
