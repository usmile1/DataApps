"""Step 1c: Verify the connector against sample_docs/."""

from connector.filesystem import FilesystemConnector


def main():
    conn = FilesystemConnector("sample_docs")

    # Discover all documents
    docs = conn.discover()
    print(f"Discovered {len(docs)} documents:\n")
    print(f"{'ID':<30} {'Type':<8} {'Size':>8}  {'Modified'}")
    print("-" * 78)
    for doc in sorted(docs, key=lambda d: d.id):
        print(
            f"{doc.id:<30} {doc.file_type:<8} {doc.size_bytes:>7}B  "
            f"{doc.last_modified:%Y-%m-%d %H:%M}"
        )

    # Fetch one document and show a preview
    print("\n--- Fetch test: customer_notes.txt ---\n")
    raw = conn.fetch("customer_notes.txt")
    print(f"  ID:   {raw.metadata.id}")
    print(f"  Type: {raw.metadata.file_type}")
    print(f"  Size: {raw.metadata.size_bytes}B")
    print(f"  Content preview:\n")
    # Show first 3 lines
    for line in raw.content.splitlines()[:3]:
        print(f"    {line}")


if __name__ == "__main__":
    main()
