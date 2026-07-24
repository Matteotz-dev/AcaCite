# AcaCite Overview

AcaCite indexes local research material into a provenance-tracked retrieval
store. Every chunk has a stable opaque citation identifier so agents can cite
the exact source chunk they inspected.

The MCP server is retrieval-only. It exposes search, source-opening metadata,
related-evidence lookup, explicit memory insertion, and health/status tools.
It does not call a local answer model.

The HTTP API also includes an optional answer endpoint. That endpoint is
separate from MCP and should be used only when local Ollama generation is
explicitly desired.

## References

Lewis, P. et al. 2020 Retrieval-augmented generation for knowledge-intensive
NLP tasks. Advances in Neural Information Processing Systems.
