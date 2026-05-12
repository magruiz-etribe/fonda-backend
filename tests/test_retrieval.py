"""Tests for retrieval.py — pure Python, no AWS needed."""
import os

os.environ.setdefault("NOVA_LITE_MODEL_ID", "test-model")
os.environ.setdefault("DDB_TABLE_NAME", "test-table")

import retrieval


class TestGetEntitiesWithVariants:
    def test_returns_list(self):
        result = retrieval.get_entities_with_variants()
        assert isinstance(result, list)

    def test_includes_mole(self):
        assert "mole" in retrieval.get_entities_with_variants()

    def test_includes_arroz(self):
        assert "arroz" in retrieval.get_entities_with_variants()

    def test_is_sorted(self):
        result = retrieval.get_entities_with_variants()
        assert result == sorted(result)

    def test_no_custom(self):
        assert "custom" not in retrieval.get_entities_with_variants()


class TestGetContextForDishes:
    def test_mole_returns_content(self):
        ctx = retrieval.get_context_for_dishes(["mole"])
        assert len(ctx) > 0
        assert "Variantes" in ctx

    def test_arroz_returns_content(self):
        ctx = retrieval.get_context_for_dishes(["arroz"])
        assert len(ctx) > 0

    def test_custom_returns_empty(self):
        assert retrieval.get_context_for_dishes(["custom"]) == ""

    def test_empty_list_returns_empty(self):
        assert retrieval.get_context_for_dishes([]) == ""

    def test_multi_dish_contains_both(self):
        ctx = retrieval.get_context_for_dishes(["mole", "arroz"])
        assert "Mole" in ctx
        assert "Arroz" in ctx

    def test_multi_dish_skips_custom(self):
        ctx = retrieval.get_context_for_dishes(["mole", "custom"])
        assert "Mole" in ctx
        assert "custom" not in ctx.lower() or "Arroz" not in ctx


class TestGetEntitiesIndex:
    def test_returns_dict(self):
        idx = retrieval.get_entities_index()
        assert isinstance(idx, dict)

    def test_mole_alias_maps_to_mole(self):
        idx = retrieval.get_entities_index()
        assert idx.get("mole") == "mole"
        assert idx.get("mole negro") == "mole"

    def test_arroz_alias_maps_to_arroz(self):
        idx = retrieval.get_entities_index()
        assert idx.get("arroz") == "arroz"
