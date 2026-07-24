"""End-to-end local Cognee ingestion and retrieval smoke test.

This script is intentionally isolated from the durable Cognee store.  The
temporary root is selected before importing either Cognee or shared_memory,
which is essential because Cognee configuration is established at import time.
"""

import asyncio
import os
from pathlib import Path
from tempfile import TemporaryDirectory


SEED_TEXT = """
Large-eddy simulation resolves the energetic turbulent motions and models the
effect of unresolved scales. A nonlinear gradient model approximates the
subfilter stress using local resolved velocity gradients. Scientific claims in
this research memory must retain links to their source documents, code
versions, and simulation cases.
""".strip()


async def run_smoke_test(expected_root: Path) -> None:
    """Run the destructive smoke workflow only inside ``expected_root``."""
    # These imports must remain after COGNEE_SHARED_ROOT is set in main().
    import shared_memory
    import cognee

    configured_root = shared_memory.SHARED_ROOT.resolve()
    if configured_root != expected_root.resolve():
        raise RuntimeError(
            "Refusing destructive smoke test: Cognee root is not the isolated "
            f"temporary directory ({configured_root} != {expected_root})"
        )

    await cognee.forget(everything=True)
    await cognee.remember(SEED_TEXT, dataset_name="smoke_test")

    results = await cognee.recall(
        query_text="What does the nonlinear gradient model use?",
        datasets=["smoke_test"],
    )

    if not results:
        raise RuntimeError("Cognee returned no results")

    rendered = []
    for result in results:
        text = getattr(result, "text", None)
        if text is None:
            text = str(result)
        rendered.append(text)

    answer = "\n".join(rendered)
    print(answer)
    if "gradient" not in answer.lower():
        raise RuntimeError("Retrieved answer did not reflect the seed document")

    print("\nCOGNEE_SMOKE_TEST_OK")


def main() -> None:
    """Create, verify, and clean up the smoke test's private Cognee root."""
    with TemporaryDirectory(prefix="cognee-smoke-") as temporary_directory:
        isolated_root = Path(temporary_directory).resolve()
        os.environ["COGNEE_SHARED_ROOT"] = str(isolated_root)
        asyncio.run(run_smoke_test(isolated_root))


if __name__ == "__main__":
    main()
