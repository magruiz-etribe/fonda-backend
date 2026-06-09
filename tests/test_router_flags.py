"""Tests for deterministic + LLM flag merge in router._compute_flags."""
import os

os.environ.setdefault("NOVA_2_LITE_MODEL_ID", "test-classifier-model")
os.environ.setdefault("NOVA_PRO_MODEL_ID", "test-generator-model")
os.environ.setdefault("DDB_TABLE_NAME", "test-table")

from unittest.mock import patch

import router


_LLM_NO_ALLERGENS: dict = {
    "allergens": False,
    "allergen_triggers": [],
    "gluten_free": True,
    "gluten_triggers": [],
    "vegetarian": True,
    "vegan": False,
    "spicy_level": "none",
    "spicy_triggers": [],
}


class TestComputeFlagsMerge:
    @patch("flag_llm.compute_flags_for_dish", return_value=_LLM_NO_ALLERGENS)
    def test_huevos_revueltos_detects_egg_from_kb_defaults(self, _mock_llm):
        flags = router._compute_flags("huevos_revueltos", [], ["jamon"])
        assert "huevo" in flags["allergen_triggers"]
        assert flags["allergens"] is True

    @patch("flag_llm.compute_flags_for_dish", return_value=_LLM_NO_ALLERGENS)
    def test_enchiladas_detects_dairy_from_kb_defaults(self, _mock_llm):
        flags = router._compute_flags("enchiladas", [], ["pollo", "salsa verde"])
        assert "queso" in flags["allergen_triggers"] or "crema" in flags["allergen_triggers"]
        assert flags["allergens"] is True

    @patch("flag_llm.compute_flags_for_dish")
    def test_llm_triggers_are_preserved_when_rule_misses(self, mock_llm):
        mock_llm.return_value = {
            **_LLM_NO_ALLERGENS,
            "allergens": True,
            "allergen_triggers": ["cacahuate"],
        }
        flags = router._compute_flags("arroz", [], ["blanco"])
        assert "cacahuate" in flags["allergen_triggers"]

    @patch("flag_llm.compute_flags_for_dish", return_value=_LLM_NO_ALLERGENS)
    def test_jamon_makes_dish_not_vegetarian(self, _mock_llm):
        flags = router._compute_flags("huevos_revueltos", [], ["jamon"])
        assert flags["vegetarian"] is False
        assert flags["vegan"] is False
