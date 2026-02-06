"""
Quick smoke tests for all dharma-agent modules.
Run with: python test_modules.py
"""

import sys

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1


# ── 1. Prompts ───────────────────────────────────────────────────────────────

def test_prompts():
    from prompts import SYSTEM_PROMPT, TEMPERATURE_FACTUAL, TEMPERATURE_CREATIVE, TEMPERATURE_DEFAULT
    assert len(SYSTEM_PROMPT) > 1000, f"Prompt too short: {len(SYSTEM_PROMPT)}"
    assert "Identity" in SYSTEM_PROMPT
    assert "Anti-Hallucination" in SYSTEM_PROMPT
    assert "Methodology" in SYSTEM_PROMPT
    assert 0.0 < TEMPERATURE_FACTUAL < 1.0
    assert 0.0 < TEMPERATURE_CREATIVE < 1.0

print("\n1. Prompts")
test("System prompt loads with all sections", test_prompts)


# ── 2. Entity Database ──────────────────────────────────────────────────────

def test_entity_db_init():
    from entities import EntityDatabase
    db = EntityDatabase()
    assert db.total_count > 200, f"Too few entities: {db.total_count}"

def test_entity_verify_known():
    from entities import EntityDatabase
    db = EntityDatabase()
    t, c = db.verify_entity("Nagarjuna")
    assert t == "teacher" and c == 1.0, f"Got {t}, {c}"
    t, c = db.verify_entity("MN 26")
    assert t == "text" and c == 1.0, f"Got {t}, {c}"
    t, c = db.verify_entity("Madhyamaka")
    assert t == "school" and c == 1.0, f"Got {t}, {c}"
    t, c = db.verify_entity("Pali Canon")
    assert t == "collection" and c == 1.0, f"Got {t}, {c}"

def test_entity_verify_unknown():
    from entities import EntityDatabase
    db = EntityDatabase()
    t, c = db.verify_entity("Fabricated Teacher Name")
    assert t == "unknown" and c == 0.0, f"Got {t}, {c}"

def test_entity_case_insensitive():
    from entities import EntityDatabase
    db = EntityDatabase()
    t, c = db.verify_entity("nagarjuna")
    assert c >= 0.9, f"Case-insensitive failed: {c}"

print("\n2. Entity Database")
test("Initializes with 200+ seed entities", test_entity_db_init)
test("Verifies known entities (teacher/text/school/collection)", test_entity_verify_known)
test("Returns unknown for fabricated entities", test_entity_verify_unknown)
test("Case-insensitive matching works", test_entity_case_insensitive)


# ── 3. Verification ─────────────────────────────────────────────────────────

def test_verify_good_text():
    from verify import verify_content
    text = ("Nagarjuna argues in the Mulamadhyamakakarika that all phenomena "
            "lack inherent existence. The Dhammapada teaches that the mind "
            "is the forerunner of all actions.")
    report = verify_content(text)
    assert len(report.verified) > 0, "Should find verified entities"
    assert len(report.unverified) == 0, f"Unexpected unverified: {[m.name for m in report.unverified]}"
    assert report.overall_confidence > 0.5

def test_verify_bad_text():
    from verify import verify_content
    text = ("According to the Fake Sutra of Infinite Wisdom, the Buddha taught "
            "that all is illusion.")
    report = verify_content(text)
    # "Fake Sutra" pattern should be flagged via "Infinite Wisdom Sutra" or similar
    assert report.overall_confidence < 1.0

def test_verify_hinayana_warning():
    from verify import verify_content
    text = "The Hinayana schools interpreted this differently."
    report = verify_content(text)
    assert any("Hinayana" in w for w in report.warnings), "Should warn about Hinayana"

def test_verify_format_report():
    from verify import verify_content, format_verification_report
    text = "Nagarjuna's Mulamadhyamakakarika argues for emptiness."
    report = verify_content(text)
    formatted = format_verification_report(report)
    assert "Confidence" in formatted
    assert isinstance(formatted, str)

print("\n3. Verification")
test("Verifies text with known entities", test_verify_good_text)
test("Flags text with unknown entities", test_verify_bad_text)
test("Warns about Hinayana usage", test_verify_hinayana_warning)
test("Formats report as string", test_verify_format_report)


# ── 4. Chunking ─────────────────────────────────────────────────────────────

def test_chunk_text_basic():
    from ingest.ingest_common import chunk_text
    text = ("First paragraph about dependent origination and how all things "
            "arise in dependence on conditions.\n\n"
            "Second paragraph about the middle way and how Nagarjuna "
            "demonstrated that neither eternalism nor nihilism captures "
            "the truth of how phenomena exist.\n\n"
            "Third paragraph about emptiness and its relationship to "
            "conventional reality in the Madhyamaka system.")
    chunks = chunk_text(text)
    assert len(chunks) >= 1, f"Expected at least 1 chunk, got {len(chunks)}"
    assert all(c.text for c in chunks), "All chunks should have text"
    assert all("chunk_index" in c.metadata for c in chunks), "All chunks should have chunk_index"

def test_chunk_verses():
    from ingest.ingest_common import chunk_verses
    verses = [f"Verse {i}: Content of verse {i} about emptiness and form." for i in range(1, 21)]
    chunks = chunk_verses(verses, group_size=4)
    assert len(chunks) >= 4, f"Expected at least 4 verse groups, got {len(chunks)}"
    assert all("verse_range" in c.metadata for c in chunks)

def test_enrich_metadata():
    from ingest.ingest_common import chunk_text, enrich_metadata
    text = "A paragraph about the four noble truths and the eightfold path."
    chunks = chunk_text(text)
    if chunks:
        enriched = enrich_metadata(chunks, tradition="Theravada", text_id="DN 22",
                                   translator="Bhikkhu Bodhi", canonical_collection="Sutta Pitaka")
        assert enriched[0].metadata["tradition"] == "Theravada"
        assert enriched[0].metadata["text_id"] == "DN 22"

print("\n4. Chunking")
test("Splits text into chunks with metadata", test_chunk_text_basic)
test("Groups verses with overlap", test_chunk_verses)
test("Enriches metadata on chunks", test_enrich_metadata)


# ── 5. RAG Module ───────────────────────────────────────────────────────────

def test_rag_init():
    import os, shutil
    from rag import DharmaRAG
    test_db = os.path.join(os.path.dirname(__file__), "_test_db")
    try:
        rag = DharmaRAG(db_path=test_db)
        assert rag.collection.count() == 0
        stats = rag.get_stats()
        assert stats["total_chunks"] == 0
    finally:
        if os.path.exists(test_db):
            shutil.rmtree(test_db, ignore_errors=True)

def test_rag_index_and_retrieve():
    import os, shutil
    from rag import DharmaRAG
    from ingest.ingest_common import TextChunk
    test_db = os.path.join(os.path.dirname(__file__), "_test_db2")
    try:
        rag = DharmaRAG(db_path=test_db)
        chunks = [
            TextChunk(text="Nagarjuna's Mulamadhyamakakarika argues that all phenomena "
                          "lack inherent existence through systematic analysis of causation.",
                     metadata={"text_id": "MMK", "tradition": "Mahayana",
                              "translator": "Jay Garfield", "chunk_index": 0,
                              "canonical_collection": "Indian Shastra", "type": "verse"}),
            TextChunk(text="The Dhammapada teaches that the mind is the forerunner of all "
                          "actions. By mind the world is led, by mind the world is drawn.",
                     metadata={"text_id": "Dhp", "tradition": "Theravada",
                              "translator": "Thanissaro Bhikkhu", "chunk_index": 0,
                              "canonical_collection": "Khuddaka Nikaya", "type": "sutta"}),
        ]
        indexed = rag.index_chunks(chunks, show_progress=False)
        assert indexed == 2, f"Expected 2 indexed, got {indexed}"
        assert rag.collection.count() == 2

        # Test retrieval
        context, sources = rag.retrieve("What is emptiness in Madhyamaka?", k=2)
        assert len(sources) == 2
        assert context  # Should have formatted context

        # Test tradition filter
        context, sources = rag.retrieve("emptiness", k=2, tradition_filter="Theravada")
        assert all(s.get("tradition") == "Theravada" for s in sources)
    finally:
        if os.path.exists(test_db):
            shutil.rmtree(test_db, ignore_errors=True)

print("\n5. RAG Module (this downloads the embedding model on first run)")
test("Initializes with empty database", test_rag_init)
test("Indexes chunks and retrieves with filtering", test_rag_index_and_retrieve)


# ── 6. Main agent imports ───────────────────────────────────────────────────

def test_agent_imports():
    # Just verify the main module can be imported without errors
    import importlib
    # We can't fully import dharma-agent.py because it has a hyphen,
    # but we can verify the key imports work
    from prompts import SYSTEM_PROMPT
    from rag import DharmaRAG
    from entities import EntityDatabase
    from verify import verify_content
    assert True

print("\n6. Agent Integration")
test("All modules import successfully", test_agent_imports)


# ── Summary ──────────────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'='*50}")

if failed > 0:
    sys.exit(1)
