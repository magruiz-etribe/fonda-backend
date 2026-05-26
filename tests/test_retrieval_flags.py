"""Unit tests for ingredient collection used in flag computation."""
import os

os.environ.setdefault("NOVA_2_LITE_MODEL_ID", "test-classifier-model")
os.environ.setdefault("NOVA_PRO_MODEL_ID", "test-generator-model")
os.environ.setdefault("DDB_TABLE_NAME", "test-table")

import retrieval


class TestCollectIngredientsForFlags:
    def test_huevo_con_jamon_includes_jamon(self):
        ingredients = retrieval.collect_ingredients_for_flags(
            "huevo",
            {},
            "huevo con jamón",
        )
        assert "jamon" in ingredients
        assert "huevo" in ingredients

    def test_plain_huevo_excludes_meat(self):
        ingredients = retrieval.collect_ingredients_for_flags(
            "huevo",
            {},
            "huevo",
        )
        assert "huevo" in ingredients
        assert "jamon" not in ingredients
        assert "chorizo" not in ingredients

    def test_resolved_variant_adds_extras(self):
        ingredients = retrieval.collect_ingredients_for_flags(
            "huevo",
            {"huevo": "con_jamon"},
            "huevo",
        )
        assert "jamon" in ingredients
        assert "cebolla" in ingredients

    def test_mentioned_kb_ingredient_from_any_variant(self):
        ingredients = retrieval.collect_ingredients_for_flags(
            "huevo",
            {},
            "huevo revueltos con chorizo",
        )
        assert "chorizo" in ingredients
