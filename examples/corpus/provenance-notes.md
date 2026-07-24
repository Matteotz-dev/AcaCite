# Provenance Notes

AcaCite stores document identity, document versions, chunk hashes, source
metadata, and ingestion job records in SQLite. Vector stores can be rebuilt
from the provenance registry when the index version or embedding model changes.

Approved ingestion roots limit what the service is allowed to read. This keeps
operator mistakes from recursively indexing unrelated local directories.

## References

Johnson, J., Douze, M. & Jegou, H. 2019 Billion-scale similarity search with
GPUs. IEEE Transactions on Big Data.
