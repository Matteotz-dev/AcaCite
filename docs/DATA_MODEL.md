# Provenance data model

SQLite is the canonical ledger for document identity, immutable versions,
chunks, and citation resolution. The Phase 1 schema is in
`app/db/schema.sql`; migrations are applied transactionally by
`app.db.migrations.migrate`.

## Invariants

- A document is identified by `(dataset, canonical_uri)` using a stable UUIDv5.
- A version is identified by `(document_id, content_hash, index_version)`.
- A chunk is identified by `(document_version_id, ordinal, text_hash)`.
- Re-registering identical data is idempotent.
- `documents.content_hash` always describes `current_version_id` once a current
  version exists. Discovering changed bytes does not update it until
  `mark_version_current()` is called after successful indexing.
- Old versions remain addressable and are marked `superseded` when a new one
  becomes current.
- Public citations use opaque `[SRC:<chunk UUID>]` tokens and are resolved
  through SQLite rather than trusting model-generated paths.
- Local open paths are returned only when their resolved target lies under an
  explicitly approved ingestion root. Symlink escapes are rejected.

## Runtime initialization

Creating the repository object has no filesystem side effects. The caller must
explicitly initialize it:

```python
from app.config import get_settings
from app.db import ProvenanceRepository

settings = get_settings()
repository = ProvenanceRepository(settings.provenance_db_path)
repository.initialize()
```

Tests must always provide a temporary database path. Phase 1 does not create or
migrate the production `data/provenance.sqlite3` automatically; API startup
integration begins with the ingestion service in Phase 2.
