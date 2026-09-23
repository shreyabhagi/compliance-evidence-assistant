"""Focused regression tests for the Microeai_RAG_v2.py reliability patch.

Run with:  python -m pytest test_microeai_rag.py -q

Streamlit is replaced by a no-op stub so the app module can be imported without a browser session,
ChromaDB runs against a throwaway directory, and Ollama embedding calls are replaced by fakes.
"""
import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest


# --- Minimal Streamlit stand-in ---------------------------------------------------------------------
class _Noop:
    """Falsy, callable, usable as a context manager; every attribute is another _Noop."""

    def __call__(self, *args, **kwargs):
        return _Noop()

    def __getattr__(self, name):
        return _Noop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __bool__(self):
        return False


class _SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = value


def _install_fake_streamlit():
    fake = types.ModuleType("streamlit")
    fake.session_state = _SessionState()
    fake.sidebar = _Noop()
    fake.cache_resource = lambda func: func
    fake.slider = lambda label, lo, hi, value, *args, **kwargs: value
    fake.select_slider = lambda label, options, value, **kwargs: value

    def stop():
        raise RuntimeError("st.stop() called during import")

    fake.stop = stop
    fake.__getattr__ = lambda name: _Noop()
    sys.modules["streamlit"] = fake


_TMP_DIR = tempfile.mkdtemp(prefix="microeai_test_")
os.environ["CHROMA_DB_PATH"] = os.path.join(_TMP_DIR, "chroma")
os.environ["SAMPLE_DOCUMENT_DIR"] = os.path.join(_TMP_DIR, "no_samples")
os.environ["EMBED_BATCH_SIZE"] = "2"
_install_fake_streamlit()

_spec = importlib.util.spec_from_file_location("microeai_rag", Path(__file__).with_name("Microeai_RAG_v2.py"))
app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app)


# --- Helpers ----------------------------------------------------------------------------------------
class FakeUpload:
    def __init__(self, name):
        self.name = name


def _fake_vector(text):
    return [float(len(text) % 7 + 1), 1.0, 0.5, 0.25]


class EmbedCounter:
    def __init__(self, fail_on_batch=None):
        self.single_calls = 0
        self.batch_calls = 0
        self.fail_on_batch = fail_on_batch

    def get_embedding(self, text):
        self.single_calls += 1
        return _fake_vector(text)

    def get_embeddings(self, texts):
        self.batch_calls += 1
        if self.fail_on_batch is not None and self.batch_calls == self.fail_on_batch:
            raise RuntimeError("simulated Ollama failure")
        return [_fake_vector(t) for t in texts]


# Several sections so the document yields more than one embedding batch (EMBED_BATCH_SIZE=2).
MULTI_SECTION_TEXT = "\n".join(
    f"{i}. Section Heading {i}\nBody text for section {i} describing monitoring requirement {i}."
    for i in range(1, 7)
)


@pytest.fixture
def index_env(monkeypatch):
    session = f"test-{os.urandom(4).hex()}"
    source = "study_doc.docx"
    counter = EmbedCounter()
    monkeypatch.setattr(app, "extract_docx_text", lambda _f: MULTI_SECTION_TEXT)
    monkeypatch.setattr(app, "get_embedding", counter.get_embedding)
    monkeypatch.setattr(app, "get_embeddings", counter.get_embeddings)
    return session, source, counter


def _doc_meta(session, source):
    return app.get_indexed_document_metadata(source, session)


def _chunk_ids(session, source):
    result = app.chunks_collection.get(
        where={"$and": [{"session_id": session}, {"source": source}]}, include=[]
    )
    return result["ids"]


def _set_status(session, source, status):
    meta = dict(_doc_meta(session, source))
    meta["index_status"] = status
    app.docs_collection.update(ids=[f"{session}::{source}"], metadatas=[meta])


# --- Indexing ---------------------------------------------------------------------------------------
def test_01_complete_same_hash_skips_indexing(index_env):
    session, source, counter = index_env
    app.load_and_index_docx(FakeUpload(source), session)
    assert _doc_meta(session, source)["index_status"] == "COMPLETE"
    calls_before = (counter.single_calls, counter.batch_calls)

    app.load_and_index_docx(FakeUpload(source), session)
    assert (counter.single_calls, counter.batch_calls) == calls_before


@pytest.mark.parametrize("status", ["PENDING", "FAILED"])
def test_02_03_incomplete_same_hash_is_retried(index_env, status):
    session, source, counter = index_env
    app.load_and_index_docx(FakeUpload(source), session)
    _set_status(session, source, status)
    batch_calls_before = counter.batch_calls

    app.load_and_index_docx(FakeUpload(source), session)
    assert counter.batch_calls > batch_calls_before
    assert _doc_meta(session, source)["index_status"] == "COMPLETE"


def test_legacy_metadata_without_status_is_not_trusted(index_env):
    session, source, counter = index_env
    app.load_and_index_docx(FakeUpload(source), session)
    meta = dict(_doc_meta(session, source))
    for key in ("index_status", "expected_chunk_count", "persisted_chunk_count"):
        meta.pop(key)
    assert app.is_index_complete(meta, meta["doc_hash"]) is False


def test_04_failed_batch_prevents_complete(index_env):
    session, source, counter = index_env
    counter.fail_on_batch = 2
    app.load_and_index_docx(FakeUpload(source), session)

    meta = _doc_meta(session, source)
    assert meta["index_status"] == "FAILED"
    assert meta["persisted_chunk_count"] < meta["expected_chunk_count"]
    # Batch 1 was persisted immediately, before batch 2 failed.
    assert len(_chunk_ids(session, source)) == meta["persisted_chunk_count"] > 0
    assert app.is_index_complete(meta, meta["doc_hash"]) is False


def test_05_all_batches_succeed_marks_complete(index_env):
    session, source, counter = index_env
    app.load_and_index_docx(FakeUpload(source), session)

    meta = _doc_meta(session, source)
    assert meta["index_status"] == "COMPLETE"
    assert meta["persisted_chunk_count"] == meta["expected_chunk_count"] > 2
    assert len(_chunk_ids(session, source)) == meta["expected_chunk_count"]
    # One document-level embedding; one batch call per EMBED_BATCH_SIZE chunks.
    assert counter.single_calls == 1
    assert counter.batch_calls == -(-meta["expected_chunk_count"] // app.EMBED_BATCH_SIZE)


def test_06_retry_does_not_duplicate_chunks(index_env):
    session, source, counter = index_env
    counter.fail_on_batch = 2
    app.load_and_index_docx(FakeUpload(source), session)
    counter.fail_on_batch = None

    app.load_and_index_docx(FakeUpload(source), session)
    meta = _doc_meta(session, source)
    ids = _chunk_ids(session, source)
    assert meta["index_status"] == "COMPLETE"
    assert len(ids) == len(set(ids)) == meta["expected_chunk_count"]
    assert len(app.docs_collection.get(ids=[f"{session}::{source}"])["ids"]) == 1


# --- Citations / claim validation -------------------------------------------------------------------
SAE_CHUNK = {
    "source": "sae_report.docx",
    "section": "Event description",
    "chunk_index": 0,
    "text": "Section: Event description\nPatient PT-0042 was hospitalized with Grade 3 neutropenia "
            "on 12 March 2024. The investigator assessed the event as possibly related to study drug.",
}
POLICY_CHUNK = {
    "source": "retention_policy.docx",
    "section": "Records retention",
    "chunk_index": 0,
    "text": "Section: Records retention\nQuality records must be archived for seven years after "
            "study closure in the controlled document repository.",
}


def _context(retrieved):
    return "\n".join(chunk["text"] for chunk, _ in retrieved)


def _statuses(answer, retrieved):
    return [c["status"] for c in app.validate_claims(answer, retrieved, _context(retrieved))]


def test_07_missing_citation_is_flagged_not_auto_filled():
    retrieved = [(SAE_CHUNK, 0.9)]
    valid = {"sae_report.docx"}
    raw = "- Patient PT-0042 was hospitalized with Grade 3 neutropenia on 12 March 2024."

    final = app.finalize_answer_text(raw, valid, "What happened to the patient?")
    assert "Source:" not in final
    assert not hasattr(app, "add_missing_single_source_citations")

    checks = app.run_groundedness_checks(final, valid, retrieved, _context(retrieved))
    assert [c["status"] for c in checks["claims"]] == ["MISSING_CITATION"]
    assert "Some statements are missing inline source citations." in app.build_groundedness_warning(checks)


def test_08_citation_to_unretrieved_source_is_unverified():
    retrieved = [(SAE_CHUNK, 0.9)]
    valid = {"sae_report.docx"}
    answer = "- Patient PT-0042 was hospitalized with Grade 3 neutropenia (Source: other_report.docx)"

    checks = app.run_groundedness_checks(answer, valid, retrieved, _context(retrieved))
    assert checks["unverified_citations"] == ["other_report.docx"]
    assert checks["claims"][0]["status"] == "CITATION_MISMATCH"
    assert "One or more citations may not support the associated statement." in app.build_groundedness_warning(checks)


def test_09_citation_to_unrelated_retrieved_source_is_flagged():
    retrieved = [(SAE_CHUNK, 0.9), (POLICY_CHUNK, 0.8)]
    answer = "- Patient PT-0042 was hospitalized with Grade 3 neutropenia (Source: retention_policy.docx)"
    assert _statuses(answer, retrieved) == ["CITATION_MISMATCH"]

    # Unrelated evidence with no better-matching source is still flagged, just less specifically.
    only_policy = [(POLICY_CHUNK, 0.8)]
    assert _statuses(answer, only_policy) == ["POSSIBLY_UNSUPPORTED"]


def test_supported_claim_passes():
    retrieved = [(SAE_CHUNK, 0.9), (POLICY_CHUNK, 0.8)]
    answer = (
        "- Patient PT-0042 was hospitalized with Grade 3 neutropenia on 12 March 2024 (Source: sae_report.docx)\n"
        "- Quality records must be archived for seven years (Source: retention_policy.docx)"
    )
    assert _statuses(answer, retrieved) == ["SUPPORTED", "SUPPORTED"]
    checks = app.run_groundedness_checks(answer, {"sae_report.docx", "retention_policy.docx"}, retrieved, _context(retrieved))
    assert app.build_groundedness_warning(checks) is None


def test_unsupported_number_is_flagged():
    retrieved = [(SAE_CHUNK, 0.9)]
    answer = "- Patient PT-0042 was hospitalized with Grade 4 neutropenia on 12 March 2024 (Source: sae_report.docx)"
    claim = app.validate_claims(answer, retrieved, _context(retrieved))[0]
    assert claim["status"] == "POSSIBLY_UNSUPPORTED"
    assert "4" in claim["detail"]


# --- Terminology ------------------------------------------------------------------------------------
def _ctcae_chunk(text):
    return {"source": "protocol.docx", "section": "Safety monitoring", "chunk_index": 1, "text": text}


def test_10_unsupported_acronym_expansion_flagged():
    retrieved = [(_ctcae_chunk("Section: Safety monitoring\nAdverse events are graded using CTCAE version 4.0."), 0.9)]
    answer = ("- Adverse events are graded using Common Terminology Criteria for Adverse Events (CTCAE) "
              "version 4.0 (Source: protocol.docx)")
    assert _statuses(answer, retrieved) == ["TERMINOLOGY_TRANSFORMATION"]
    checks = app.run_groundedness_checks(answer, {"protocol.docx"}, retrieved, _context(retrieved))
    assert "Some terminology may have been expanded or transformed" in app.build_groundedness_warning(checks)


def test_11_expansion_present_in_source_not_flagged():
    retrieved = [(_ctcae_chunk(
        "Section: Safety monitoring\nAdverse events are graded using Common Terminology Criteria for "
        "Adverse Events (CTCAE) version 4.0."), 0.9)]
    answer = ("- Adverse events are graded using Common Terminology Criteria for Adverse Events (CTCAE) "
              "version 4.0 (Source: protocol.docx)")
    assert _statuses(answer, retrieved) == ["SUPPORTED"]
    checks = app.run_groundedness_checks(answer, {"protocol.docx"}, retrieved, _context(retrieved))
    assert checks["unverified_acronyms"] == []


# --- Parentheses ------------------------------------------------------------------------------------
def test_12_parenthetical_content_survives_polish():
    text = ("- Neutropenia was reported as serious (Grade 3) per CTCAE (Version 4.0) in Arm A "
            "(Arm A) (Source: sae_report.docx)")
    polished = app.polish_final_answer(text)
    for kept in ("(Grade 3)", "(Version 4.0)", "(Arm A)", "(Source: sae_report.docx)"):
        assert kept in polished


# --- Document selection -----------------------------------------------------------------------------
def test_13_end_of_document_terms_reach_embedding_text():
    filler = "General administrative background text for the study. " * 200
    full_text = "TRIAL MASTER FILE\n" + filler + "\nPharmacovigilance escalation matrix applies to all sites."
    assert "Pharmacovigilance escalation matrix" not in full_text[: app.DOCUMENT_PREVIEW_LENGTH]

    sample = app.build_document_embedding_text(full_text)
    assert "Pharmacovigilance escalation matrix" in sample
    assert sample.startswith("TRIAL MASTER FILE")
    assert len(sample) <= app.DOCUMENT_PREVIEW_LENGTH + 20  # separators only


def test_short_document_embedding_text_unchanged():
    assert app.build_document_embedding_text("Short doc.") == "Short doc."


# --- Routing ----------------------------------------------------------------------------------------
def test_14_protocol_detected_from_content_not_filename():
    text = ("CLINICAL TRIAL PROTOCOL\nProtocol Number: XYZ-101\nPROTOCOL SYNOPSIS\n"
            "This study evaluates drug X in adults.")
    assert app.infer_document_type("study_doc_A.docx", text) == "clinical_trial_protocol"


def test_sae_report_mentioning_protocol_number_is_sae():
    text = "SERIOUS ADVERSE EVENT REPORT\nProtocol Number: XYZ-101\nSAE Number: 7"
    assert app.infer_document_type("report.docx", text) == "sae_report"


def test_filename_fallback_and_other():
    assert app.infer_document_type("SOP-12 cleaning.docx", "Some text here.") == "sop"
    assert app.infer_document_type("notes.docx", "Some text here.") == "other"


def test_scope_uses_metadata_and_keeps_filename_matches():
    sources = ["study_doc_A.docx", "old_protocol.docx", "notes.docx"]
    types_ = {"study_doc_A.docx": "clinical_trial_protocol", "notes.docx": "other"}
    scoped = app.apply_query_source_scope("What protocol safety monitoring is described?", sources, [], types_)
    assert scoped == ["study_doc_A.docx", "old_protocol.docx"]
    # No confident match -> falls back to the selected documents, not an empty list.
    assert app.apply_query_source_scope("What does the SOP say?", sources, ["notes.docx"], types_) == ["notes.docx"]


# --- Only COMPLETE documents are answer evidence ----------------------------------------------------
def _store_document(session, source, status, expected, persisted, vector):
    app.docs_collection.upsert(
        ids=[f"{session}::{source}"],
        documents=[f"{source} preview"],
        embeddings=[vector],
        metadatas=[{
            "source": source, "session_id": session, "doc_hash": "h", "index_status": status,
            "expected_chunk_count": expected, "persisted_chunk_count": persisted,
        }],
    )
    app.chunks_collection.upsert(
        ids=[f"{session}::{source}__{i}" for i in range(persisted)],
        documents=[f"Section: Safety\n{source} safety monitoring evidence {i}" for i in range(persisted)],
        embeddings=[vector] * persisted,
        metadatas=[
            {"source": source, "session_id": session, "doc_hash": "h", "section": "Safety", "chunk_index": i}
            for i in range(persisted)
        ],
    )


def test_failed_document_chunks_are_not_answer_evidence():
    session = f"test-{os.urandom(4).hex()}"
    query_vec = [1.0, 0.0, 0.0, 0.0]
    # The FAILED document's chunks are an exact vector match for the query, so only the status filter can exclude them.
    _store_document(session, "failed.docx", "FAILED", expected=10, persisted=5, vector=query_vec)
    _store_document(session, "complete.docx", "COMPLETE", expected=3, persisted=3, vector=[0.9, 0.3, 0.0, 0.0])
    assert len(_chunk_ids(session, "failed.docx")) == 5  # still stored for recovery

    selected = app.choose_relevant_documents("safety monitoring", session, query_embedding=query_vec, threshold=0.0)
    assert selected == ["complete.docx"]

    for filters in (None, ["failed.docx", "complete.docx"]):
        retrieved = app.retrieve("safety monitoring", session, top_n=6, source_filters=filters, query_embedding=query_vec)
        sources = {chunk["source"] for chunk, _ in retrieved}
        assert sources == {"complete.docx"}
        assert len(retrieved) == 3  # COMPLETE document remains fully retrievable

    # Scoped to only the FAILED document -> no evidence rather than its partial chunks.
    assert app.retrieve("safety monitoring", session, source_filters=["failed.docx"], query_embedding=query_vec) == []


# --- Safe deterministic repair (validate -> repair -> re-validate) ----------------------------------
# Test H (COMPLETE-only retrieval) is test_failed_document_chunks_are_not_answer_evidence above.
def _chunk(source, text, index=0):
    return {"source": source, "section": "Narrative", "chunk_index": index, "text": f"Section: Narrative\n{text}"}


def _repair_and_check(answer, retrieved):
    """Mirrors the main flow: repair, then re-validate the repaired text."""
    context = _context(retrieved)
    valid = {chunk["source"] for chunk, _ in retrieved}
    repaired = app.apply_safe_repairs(answer, retrieved, context)
    checks = app.run_groundedness_checks(repaired, valid, retrieved, context)
    return repaired, checks, app.build_groundedness_warning(checks)


def test_A_supported_uncited_claim_gets_its_citation():
    retrieved = [(_chunk("trial.docx", "The participant developed acute urinary retention approximately "
                                       "12 days after HDR treatment."), 0.9)]
    answer = "- The participant developed acute urinary retention approximately 12 days after HDR treatment."
    repaired, checks, warning = _repair_and_check(answer, retrieved)
    assert repaired == ("- The participant developed acute urinary retention approximately 12 days after "
                        "HDR treatment. (Source: trial.docx)")
    assert [c["status"] for c in checks["claims"]] == ["SUPPORTED"]
    assert warning is None


def test_B_single_source_is_not_blindly_cited():
    retrieved = [(_chunk("trial.docx", "The participant developed urinary retention."), 0.9)]
    answer = "- The participant developed liver failure."
    repaired, checks, warning = _repair_and_check(answer, retrieved)
    assert repaired == answer  # only one source retrieved, but the claim does not match it
    assert [c["status"] for c in checks["claims"]] == ["MISSING_CITATION"]
    assert "Some statements are missing inline source citations." in warning


def test_B_unsupported_number_is_not_cited():
    retrieved = [(_chunk("trial.docx", "The participant developed acute urinary retention approximately "
                                       "12 days after HDR treatment."), 0.9)]
    answer = "- The participant developed acute urinary retention approximately 20 days after HDR treatment."
    repaired, checks, _ = _repair_and_check(answer, retrieved)
    assert repaired == answer
    assert checks["claims"][0]["status"] == "MISSING_CITATION"


def test_C_ambiguous_multi_source_support_is_not_guessed():
    text = "Serious adverse events must be reported to the sponsor within 24 hours of site awareness."
    retrieved = [(_chunk("protocol.docx", text), 0.9), (_chunk("sae.docx", text), 0.8)]
    answer = "- Serious adverse events must be reported to the sponsor within 24 hours of site awareness."
    repaired, checks, warning = _repair_and_check(answer, retrieved)
    assert repaired == answer
    assert checks["claims"][0]["status"] == "MISSING_CITATION"
    assert warning is not None


def test_C_unique_source_among_several_is_cited():
    retrieved = [
        (_chunk("protocol.docx", "Adverse events are graded using CTCAE version 4.0."), 0.9),
        (_chunk("sae.docx", "The participant developed acute urinary retention 12 days after HDR treatment."), 0.8),
    ]
    answer = "- The participant developed acute urinary retention 12 days after HDR treatment."
    repaired, checks, _ = _repair_and_check(answer, retrieved)
    assert repaired.endswith("(Source: sae.docx)")
    assert checks["claims"][0]["status"] == "SUPPORTED"


def test_D_unsupported_acronym_expansion_is_reverted():
    retrieved = [(_chunk("protocol.docx", "Adverse events are graded using CTCAE version 4.0."), 0.9)]
    answer = ("- Adverse events are graded using Common Terminology Criteria for Adverse Events (CTCAE) "
              "version 4.0. (Source: protocol.docx)")
    repaired, checks, warning = _repair_and_check(answer, retrieved)
    assert repaired == "- Adverse events are graded using CTCAE version 4.0. (Source: protocol.docx)"
    assert checks["unverified_acronyms"] == []
    assert [c["status"] for c in checks["claims"]] == ["SUPPORTED"]
    assert warning is None


def test_D_expansion_and_missing_citation_both_repaired():
    retrieved = [(_chunk("protocol.docx", "Adverse events are graded using CTCAE version 4.0."), 0.9)]
    answer = "- Adverse events are graded using Common Terminology Criteria for Adverse Events (CTCAE) version 4.0."
    repaired, checks, warning = _repair_and_check(answer, retrieved)
    assert repaired == "- Adverse events are graded using CTCAE version 4.0. (Source: protocol.docx)"
    assert warning is None


def test_E_supported_acronym_expansion_is_preserved():
    evidence = "Adverse events are graded using Common Terminology Criteria for Adverse Events (CTCAE) version 4.0."
    retrieved = [(_chunk("protocol.docx", evidence), 0.9)]
    answer = ("- Adverse events are graded using Common Terminology Criteria for Adverse Events (CTCAE) "
              "version 4.0. (Source: protocol.docx)")
    repaired, checks, warning = _repair_and_check(answer, retrieved)
    assert repaired == answer
    assert checks["unverified_acronyms"] == []
    assert warning is None


def test_F_legitimate_parentheses_are_preserved():
    retrieved = [(_chunk("sae.docx", "Neutropenia was Grade 3 (severe) in Arm A (monotherapy), graded per CTCAE."), 0.9)]
    answer = "- Neutropenia was Grade 3 (severe) in Arm A (monotherapy), graded per CTCAE (CTCAE). (Source: sae.docx)"
    polished = app.polish_final_answer(answer)
    repaired, _, _ = _repair_and_check(polished, retrieved)
    for kept in ("Grade 3 (severe)", "Arm A (monotherapy)", "(CTCAE)", "(Source: sae.docx)"):
        assert kept in repaired


def test_F_non_initialism_acronym_left_unchanged_and_warned():
    # "Serious Event Report (SAE)" initials do not spell SAE: not mechanically reversible, so no edit.
    retrieved = [(_chunk("sae.docx", "The SAE was reported to the sponsor."), 0.9)]
    answer = "- The Serious Event Report (SAE) was reported to the sponsor. (Source: sae.docx)"
    repaired, _, _ = _repair_and_check(answer, retrieved)
    assert repaired == answer


def test_G_citation_mismatch_is_not_rewritten():
    retrieved = [
        (_chunk("protocol.docx", "Quality records must be archived for seven years after study closure."), 0.9),
        (_chunk("sae.docx", "Patient PT-0042 was hospitalized with Grade 3 neutropenia on 12 March 2024."), 0.8),
    ]
    answer = "- Patient PT-0042 was hospitalized with Grade 3 neutropenia on 12 March 2024. (Source: protocol.docx)"
    repaired, checks, warning = _repair_and_check(answer, retrieved)
    assert repaired == answer
    assert [c["status"] for c in checks["claims"]] == ["CITATION_MISMATCH"]
    assert "One or more citations may not support the associated statement." in warning


def test_unsupported_regulatory_reference_is_not_cited():
    retrieved = [(_chunk("sae.docx", "The event was reported to the sponsor within 24 hours per 21 CFR 312.32."), 0.9)]
    answer = "- The event was reported to the sponsor within 24 hours per 21 CFR 314.80."
    repaired, checks, _ = _repair_and_check(answer, retrieved)
    assert repaired == answer
    assert checks["unverified_terms"]


def test_app_version_bumped():
    assert app.APP_VERSION == "2.2.0"
