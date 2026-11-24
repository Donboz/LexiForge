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
    is_junk_meaning,
    is_protected_headword,
    normalize_domain,
    sanitize_article_and_gender,
)
from handlers.fallback_chain import parse_json_response  # noqa: E402


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
