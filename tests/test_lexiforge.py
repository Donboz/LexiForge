#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for LexiForge lexical filtering, sanitization, response parsing, and CLI tools."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import psycopg2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from handlers.lexical_filter import (  # noqa: E402
    SUPPORTED_LANGS,
    classify_headword,
    clean_definitions,
    detect_language_anomaly,
    is_junk_meaning,
    is_protected_headword,
    normalize_domain,
    sanitize_article_and_gender,
    sanitize_grammar_fields,
)
from handlers.fallback_chain import parse_json_response  # noqa: E402
from scripts.nim_sql_sync import (  # noqa: E402
    NimPool,
    _apply_stage2_guards,
    _run_tx_with_retry,
    parse_glossa_pdf_text,
    run_pair_sweep_engine,
)


class TestLexicalFilter(unittest.TestCase):
    def test_supported_langs_cover_all_12_glossa_languages(self):
        self.assertEqual(
            set(SUPPORTED_LANGS),
            {"DE", "TR", "EN", "FR", "ES", "IT", "PT", "RO", "LA", "RU", "SR", "MK"},
        )

    def test_classify_headword_deletes_junk(self):
        junk_cases = [
            ("11th", "EN", "NOUN", None),
            ("11.", "DE", "NUMERAL", None),
            ("1st", "EN", None, None),
            ("1000th", "EN", None, None),
            ("1337", "EN", None, None),
            ("136brodeln –B ruc h", "DE", None, None),
            ("152dampfen –d ann", "DE", None, None),
            ("Seite 14", "DE", None, None),
            ("u", "EN", "NOUN", [{"meaning": "u"}]),
            ("D", "DE", "NOUN", [{"meaning": "D"}]),
            ("A", "DE", "NOUN", [{"meaning": "I"}, {"meaning": "M"}, {"meaning": "A"}]),
            ("0 seine Freundlichkeit ist nur M.", "DE", None, None),
        ]
        for text, lang, pos, defs in junk_cases:
            action, _cleaned, reasons = classify_headword(text, lang=lang, pos=pos, definitions=defs)
            self.assertEqual(action, "delete", f"Expected delete for {text!r}, got {action} ({reasons})")

    def test_classify_headword_preserves_valid_single_chars_and_slashes(self):
        valid_cases = [
            ("24/7", "EN", "ADVERB", [{"meaning": "7/24"}]),
            ("9/11", "EN", "NOUN", [{"meaning": "11 eylül"}]),
            ("a", "EN", None, [{"meaning": "bir"}, {"meaning": "a"}]),
            ("I", "EN", None, [{"meaning": "ben"}, {"meaning": "i"}]),
            ("O", "EN", "PARTICLE", [{"meaning": "o"}, {"meaning": "ey"}]),
            ("B", "DE", "NOUN", [{"meaning": "B-flat"}, {"meaning": "bes-majeur"}]),
        ]
        for text, lang, pos, defs in valid_cases:
            action, _cleaned, reasons = classify_headword(text, lang=lang, pos=pos, definitions=defs)
            self.assertEqual(action, "keep", f"Expected keep for {text!r}, got {action} ({reasons})")

    def test_classify_headword_cleans_glued_ocr_and_notes(self):
        action, cleaned, _ = classify_headword("107Cam elması", lang="TR")
        self.assertEqual(action, "clean")
        self.assertEqual(cleaned, "Cam elması")

        action, cleaned, _ = classify_headword("(0. PI.) Ehrenhaftigkeit", lang="DE")
        self.assertEqual(action, "clean")
        self.assertEqual(cleaned, "Ehrenhaftigkeit")

    def test_classify_headword_preserves_valid_digit_compounds(self):
        valid_terms = [
            ("1,2-Butadien", "DE"),
            ("11. September", "DE"),
            ("10-Cent-Münze", "DE"),
            ("12-seitig", "DE"),
            ("15 minutes of fame", "EN"),
            ("14K", "EN"),
            ("16mo", "EN"),
            ("1-Hexanol", "DE"),
            ("1-Pentanol", "DE"),
            ("1-Propanol", "DE"),
            ("1 Uhr", "TR"),
            ("12 Uhr Mittag/Mitternacht", "TR"),
        ]
        for text, lang in valid_terms:
            action, cleaned, reasons = classify_headword(text, lang=lang)
            self.assertEqual(action, "keep", f"Expected keep for {text!r}, got {action} ({reasons})")
            self.assertEqual(cleaned, text)
            self.assertTrue(is_protected_headword(text), f"Expected is_protected_headword({text!r}) == True")

    def test_clean_definitions_and_is_junk_meaning(self):
        self.assertTrue(is_junk_meaning("11.", "11th"))
        self.assertTrue(is_junk_meaning("11'inci", "11."))
        self.assertTrue(is_junk_meaning("Haus", "haus"))
        self.assertFalse(is_junk_meaning("ev", "Haus"))

        kept, stats = clean_definitions(
            "Haus",
            [
                {"meaning": "11.", "domain": "GENERAL"},
                {"meaning": "11'inci", "domain": "GENERAL"},
                {"meaning": "Haus", "domain": "GENERAL"},
                {"meaning": "ev", "domain": "architecture"},
                {"meaning": "ev", "domain": "ARCHITECTURE"},
                {"meaning": "bina", "domain": "SPACE"},
            ],
        )
        self.assertEqual(stats["removed"], 4)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0], {"meaning": "ev", "domain": "ARCHITECTURE"})
        self.assertEqual(kept[1], {"meaning": "bina", "domain": "ASTRONOMY"})

    def test_normalize_domain_all_and_synonyms(self):
        self.assertEqual(normalize_domain("astronomy"), "ASTRONOMY")
        self.assertEqual(normalize_domain("military"), "MILITARY")
        self.assertEqual(normalize_domain("cosmos"), "ASTRONOMY")
        self.assertEqual(normalize_domain("warfare"), "MILITARY")
        self.assertEqual(normalize_domain("unknown_weird_domain"), "GENERAL")

    def test_sanitize_article_and_gender(self):
        # TR/EN strip spurious articles & genders
        tr_term = sanitize_article_and_gender(
            {"text": "masa", "lang": "TR", "partOfSpeech": "NOUN", "article": "die", "gender": "FEMININE"}
        )
        self.assertIsNone(tr_term.get("article"))
        self.assertIsNone(tr_term.get("gender"))

        en_term = sanitize_article_and_gender(
            {"text": "table", "lang": "EN", "partOfSpeech": "NOUN", "article": "der", "gender": "MASCULINE"}
        )
        self.assertIsNone(en_term.get("article"))
        self.assertIsNone(en_term.get("gender"))

        # DE normalizes article -> gender and strips POS garbage in article
        de_noun = sanitize_article_and_gender(
            {"text": "Tisch", "lang": "DE", "partOfSpeech": "NOUN", "article": "DER", "gender": "NEUTER"}
        )
        self.assertEqual(de_noun["article"], "der")
        self.assertEqual(de_noun["gender"], "MASCULINE")

        de_pos_garbage = sanitize_article_and_gender(
            {"text": "Haus", "lang": "DE", "partOfSpeech": "NOUN", "article": "n"}
        )
        self.assertIsNone(de_pos_garbage.get("article"))

        # FR/PT/RO preserve valid articles/genders
        fr_noun = sanitize_article_and_gender(
            {"text": "maison", "lang": "FR", "partOfSpeech": "NOUN", "article": "la"}
        )
        self.assertEqual(fr_noun["article"], "la")
        self.assertEqual(fr_noun["gender"], "FEMININE")

        pt_noun = sanitize_article_and_gender(
            {"text": "livro", "lang": "PT", "partOfSpeech": "NOUN", "article": "o"}
        )
        self.assertEqual(pt_noun["article"], "o")
        self.assertEqual(pt_noun["gender"], "MASCULINE")

        ro_noun = sanitize_article_and_gender(
            {"text": "caiet", "lang": "RO", "partOfSpeech": "NOUN", "article": "un", "gender": "AMBIGENERIC"}
        )
        self.assertEqual(ro_noun["article"], "un")
        self.assertEqual(ro_noun["gender"], "AMBIGENERIC")

    def test_detect_language_anomaly_all_rules(self):
        self.assertIn("cyrillic_in_non_slavic", detect_language_anomaly("дом", "DE"))
        self.assertIn("missing_cyrillic_script", detect_language_anomaly("dom", "RU"))
        self.assertIn("turkish_chars_in_foreign_term", detect_language_anomaly("yağmur", "DE"))
        self.assertIn("german_eszett_in_non_de", detect_language_anomaly("Straße", "TR"))
        self.assertIn("article_on_genderless_lang", detect_language_anomaly("ev", "TR", article="der"))
        self.assertIn("pos_garbage_in_article", detect_language_anomaly("Haus", "DE", article="f"))
        self.assertIn(
            "same_src_tgt_lang",
            detect_language_anomaly("masa", "TR", definitions=[{"meaning": "table", "language": "TR"}]),
        )
        self.assertEqual(
            detect_language_anomaly("Haus", "DE", article="das", definitions=[{"meaning": "ev", "language": "TR"}]),
            [],
        )

    def test_sanitize_grammar_fields_12_languages(self):
        ru_verb = sanitize_grammar_fields({
            "text": "читать",
            "lang": "RU",
            "partOfSpeech": "VERB",
            "transliteration": "chitat",
            "aspect": "IMPERFECTIVE",
            "aspectPair": "прочитать",
            "conjugationClass": "I",
            "article": "der",
        })
        self.assertEqual(ru_verb["script"], "CYRILLIC")
        self.assertEqual(ru_verb["transliteration"], "chitat")
        self.assertEqual(ru_verb["aspect"], "IMPERFECTIVE")
        self.assertIsNone(ru_verb["article"])

        ro_noun = sanitize_grammar_fields({
            "text": "caiet",
            "lang": "RO",
            "partOfSpeech": "NOUN",
            "article": "un",
            "gender": "AMBIGENERIC",
            "definiteForm": "caietul",
        })
        self.assertEqual(ro_noun["definiteForm"], "caietul")
        self.assertEqual(ro_noun["gender"], "AMBIGENERIC")

        tr_noun = sanitize_grammar_fields({
            "text": "kitap",
            "lang": "TR",
            "partOfSpeech": "NOUN",
            "article": "das",
            "vowelHarmony": "back/unrounded; p->b",
        })
        self.assertIsNone(tr_noun["article"])
        self.assertEqual(tr_noun["vowelHarmony"], "back/unrounded; p->b")

        de_verb = sanitize_grammar_fields({
            "text": "gehen",
            "lang": "DE",
            "partOfSpeech": "VERB",
            "auxiliary": "SEIN",
            "level": "A1",
            "secondaryDomains": ["Transport", "transport", "GENERAL"],
            "primaryDomain": "GENERAL",
        })
        self.assertEqual(de_verb["auxiliary"], "sein")
        self.assertEqual(de_verb["secondaryDomains"], ["TRANSPORT"])


class TestNimPoolAndPdfParser(unittest.TestCase):
    def test_nim_pool_builds_and_rotates_10_unique_channels(self):
        pool = NimPool(api_keys=["k1", "k2"])
        pool.models = [
            {"model": f"m{i}", "ok": True, "latency_ms": 100.0 + i, "thinking_kw": False}
            for i in range(5)
        ]
        pool._build_channels()
        self.assertEqual(len(pool.channels), 10)
        first_picks = [pool.pick_channel_order()[0]["id"] for _ in range(10)]
        self.assertEqual(len(set(first_picks)), 10)

    def test_parse_glossa_pdf_text_multi_chapter(self):
        sample_pdf = """
BÖLÜM

Almanca → İngilizce
Bu bölüm kayıtlı ve onaylı 2 adet terim içerir.

Glossa Büyük Sözlük 4

#
11th
[Genel]

1. 11th

das Haus
(n.) [Mimarlık]

1. house
Örn: Das Haus ist groß. (The house is big.)

BÖLÜM

İngilizce → İspanyolca
Bu bölüm kayıtlı ve onaylı 1 adet terim içerir.

Glossa Büyük Sözlük 2413

E
book
(n.) [Eğitim]

1. libro

BÖLÜM

Türkçe → Rusça
Bu bölüm kayıtlı ve onaylı 1 adet terim içerir.

Glossa Büyük Sözlük 9

E
elma
(n.) [Gastronomi]

1. яблоко
"""
        parsed = parse_glossa_pdf_text(sample_pdf)
        self.assertEqual(len(parsed), 4)
        self.assertEqual(parsed[0]["pair"], "DE->EN")
        self.assertEqual(parsed[0]["text"], "11th")
        self.assertEqual(parsed[1]["text"], "Haus")
        self.assertEqual(parsed[1]["article"], "das")
        self.assertEqual(parsed[1]["partOfSpeech"], "NOUN")
        self.assertEqual(parsed[1]["primaryDomain"], "ARCHITECTURE")
        self.assertEqual(parsed[1]["examples"][0]["translation"], "The house is big.")
        self.assertEqual(parsed[2]["pair"], "EN->ES")
        self.assertEqual(parsed[2]["text"], "book")
        self.assertEqual(parsed[2]["primaryDomain"], "EDUCATION")
        self.assertEqual(parsed[3]["pair"], "TR->RU")
        self.assertEqual(parsed[3]["text"], "elma")
        self.assertEqual(parsed[3]["definitions"][0]["meaning"], "яблоко")


class TestResponseParserAndCLI(unittest.TestCase):
    def test_parse_json_response_strips_think_and_fences(self):
        raw = "<think>\nReasoning step...\n</think>\n```json\n{\"items\": [{\"id\": 1, \"action\": \"keep\"}]}\n```"
        parsed = parse_json_response(raw)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed["items"][0]["action"], "keep")

    def test_clean_meanings_cli_dry_run(self):
        sample = [
            {
                "text": "Haus",
                "article": "das",
                "definitions": [
                    {"meaning": "ev; konut; ev", "domain": "GENERAL"},
                    {"meaning": "11.", "domain": "GENERAL"},
                ],
            }
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(sample, tf, ensure_ascii=False)
            tmp_path = tf.name

        try:
            proc = subprocess.run(
                [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "clean_meanings.py"), "--file", tmp_path, "--dry-run"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
            self.assertIn("Dry-run", proc.stdout)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestSweepAndGuards(unittest.TestCase):
    def test_apply_stage2_guards_protects_valid_and_clean_hint_terms(self):
        # 1. Protected headword cannot be deleted or altered
        g1 = _apply_stage2_guards(
            {"text": "11. September", "lang": "DE", "partOfSpeech": "NOUN", "definitions": [{"meaning": "11 Eylül"}]},
            {"action": "DELETE", "verdict": "DELETE", "text": None, "reason": "ordinal fragment"},
        )
        self.assertEqual(g1["action"], "FIX")
        self.assertEqual(g1["text"], "11. September")

        # 2. Chemical locant cannot be stripped or deleted
        g2 = _apply_stage2_guards(
            {"text": "1-Propanol", "lang": "DE", "partOfSpeech": "NOUN", "definitions": [{"meaning": "1-propanol"}]},
            {"action": "FIX", "text": "Propanol"},
        )
        self.assertEqual(g2["text"], "1-Propanol")

        # 3. clean_hint overrides DELETE
        g3 = _apply_stage2_guards(
            {"text": "150daher", "clean_hint": "daher", "lang": "DE", "definitions": [{"meaning": "therefore"}]},
            {"action": "DELETE", "text": None},
        )
        self.assertEqual(g3["action"], "FIX")
        self.assertEqual(g3["text"], "daher")

        # 4. Pure junk ("11th") IS allowed to be deleted
        g4 = _apply_stage2_guards(
            {"text": "11th", "lang": "EN", "definitions": [{"meaning": "11th"}]},
            {"action": "DELETE", "reason": "ordinal"},
        )
        self.assertEqual(g4["action"], "DELETE")

    def test_run_tx_with_retry_handles_deadlock(self):
        attempts = {"count": 0}
        mock_conn = mock.MagicMock()

        def flaky_tx(_conn, dry_run=False):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise psycopg2.errors.DeadlockDetected("deadlock detected")
            return "updated"

        res = _run_tx_with_retry(mock_conn, flaky_tx, dry_run=False)
        self.assertEqual(res, "updated")
        self.assertEqual(attempts["count"], 2)
        self.assertEqual(mock_conn.rollback.call_count, 1)

    def test_run_pair_sweep_engine_12_pairs_unlimited_resumable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "state.json")
            jsonl_file = os.path.join(tmpdir, "audit.jsonl")

            # Build 12 pairs, each with 3 rows (2 chunks when chunk_size=2): 1 junk + 2 real terms
            pair_defs = [(f"L{i}", "TR") for i in range(12)]
            synthetic_db = {}
            for src, tgt in pair_defs:
                rows = [
                    {
                        "termId": f"{src}-{tgt}-1",
                        "auditKey": f"{src}-{tgt}-1@{tgt}",
                        "inDb": True,
                        "text": "11th",
                        "lang": "EN",
                        "targetLang": tgt,
                        "pair": f"{src}->{tgt}",
                        "partOfSpeech": None,
                        "gender": None,
                        "article": None,
                        "plural": None,
                        "level": "UNKNOWN",
                        "primaryDomain": "GENERAL",
                        "definitions": [{"id": f"d-{src}-1", "meaning": "11th", "language": tgt, "domain": "GENERAL", "register": "NEUTRAL"}],
                    },
                    {
                        "termId": f"{src}-{tgt}-2",
                        "auditKey": f"{src}-{tgt}-2@{tgt}",
                        "inDb": True,
                        "text": f"word_{src}_a",
                        "lang": "DE",
                        "targetLang": tgt,
                        "pair": f"{src}->{tgt}",
                        "partOfSpeech": "NOUN",
                        "gender": "NEUTER",
                        "article": "das",
                        "plural": None,
                        "level": "B1",
                        "primaryDomain": "GENERAL",
                        "definitions": [{"id": f"d-{src}-2", "meaning": "kelime a", "language": tgt, "domain": "GENERAL", "register": "NEUTRAL"}],
                    },
                    {
                        "termId": f"{src}-{tgt}-3",
                        "auditKey": f"{src}-{tgt}-3@{tgt}",
                        "inDb": True,
                        "text": f"word_{src}_b",
                        "lang": "DE",
                        "targetLang": tgt,
                        "pair": f"{src}->{tgt}",
                        "partOfSpeech": "VERB",
                        "gender": None,
                        "article": None,
                        "plural": None,
                        "level": "B2",
                        "primaryDomain": "TECHNICAL",
                        "definitions": [{"id": f"d-{src}-3", "meaning": "kelime b", "language": tgt, "domain": "TECHNICAL", "register": "NEUTRAL"}],
                    },
                ]
                synthetic_db[(src, tgt)] = rows

            def fake_discover(_conn, lang_pair_filter=None):
                return [(s, t, 3) for s, t in pair_defs]

            def fake_fetch_chunk(_conn, src_lang, tgt_lang, last_id="", chunk_size=50, recheck=False):
                items = [r for r in synthetic_db[(src_lang, tgt_lang)] if r["termId"] > (last_id or "")]
                return [dict(x) for x in items[:chunk_size]]

            class FakeNim:
                channels = [{"id": "mock#k1", "ok": 0, "err": 0, "rl429": 0, "total_ms": 0.0}]

                def chat_json(self, _sys, user_prompt):
                    payload = json.loads(user_prompt.split("\n", 1)[1])
                    resp = [
                        {
                            "termId": x["termId"],
                            "verdict": "CORRECT",
                            "action": "KEEP",
                            "langValid": True,
                            "grammarValid": True,
                            "issues": [],
                            "text": x["text"],
                            "lang": x["lang"],
                            "partOfSpeech": x.get("partOfSpeech") or "NOUN",
                            "level": x.get("level") or "B1",
                            "primaryDomain": x.get("primaryDomain") or "GENERAL",
                            "definitions": x.get("definitions") or [],
                        }
                        for x in payload
                    ]
                    return resp, "mock#k1"

                def format_channel_stats(self):
                    return "mock stats"

            args = argparse.Namespace(
                source="db",
                mode="sweep",
                chunk_size=2,
                batch_size=2,
                workers=4,
                limit=0,
                purge_junk=True,
                dry_run=False,
                recheck=False,
                reset_state=False,
                state_file=state_file,
                jsonl_log=jsonl_file,
                report_file=None,
            )

            with (
                mock.patch("scripts.nim_sql_sync.psycopg2.connect", side_effect=lambda *_a, **_kw: mock.MagicMock()),
                mock.patch("scripts.nim_sql_sync.discover_db_lang_pairs", side_effect=fake_discover),
                mock.patch("scripts.nim_sql_sync.fetch_pair_chunk", side_effect=fake_fetch_chunk),
                mock.patch("scripts.nim_sql_sync.purge_junk_row_tx", return_value="deleted:ordinal_or_bare_number"),
                mock.patch("scripts.nim_sql_sync.apply_fix_row_tx", return_value="updated"),
            ):
                rc = run_pair_sweep_engine(args, "postgres://mock", FakeNim(), None, 0.0)

            self.assertEqual(rc, 0)
            state_data = json.loads(open(state_file, encoding="utf-8").read())
            self.assertEqual(len(state_data["pairs"]), 12)
            for src, tgt in pair_defs:
                st = state_data["pairs"][f"{src}->{tgt}"]
                self.assertTrue(st["done"])
                self.assertEqual(st["deleted"], 1)
                self.assertEqual(st["kept"], 2)

            lines = open(jsonl_file, encoding="utf-8").read().splitlines()
            self.assertEqual(len(lines), 36)


if __name__ == "__main__":
    unittest.main()
