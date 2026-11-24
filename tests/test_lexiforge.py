#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for LexiForge lexical filtering, sanitization, response parsing, and CLI tools."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

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
from scripts.nim_sql_sync import NimPool, parse_glossa_pdf_text  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
