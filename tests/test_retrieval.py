"""Tests for retrieval.py — pure Python, no AWS needed."""
import os

os.environ.setdefault("NOVA_2_LITE_MODEL_ID", "test-classifier-model")
os.environ.setdefault("NOVA_PRO_MODEL_ID", "test-generator-model")
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


class TestGetDishData:
    def test_mole_returns_dict(self):
        data = retrieval.get_dish_data("mole")
        assert isinstance(data, dict)

    def test_mole_has_canonical_name(self):
        assert retrieval.get_dish_data("mole")["canonical_name"] == "mole"

    def test_mole_has_four_variants(self):
        variants = retrieval.get_dish_data("mole")["variants"]
        assert set(variants.keys()) >= {"negro", "poblano", "verde", "rojo"}

    def test_mole_negro_has_expected_ingredients(self):
        negro = retrieval.get_dish_data("mole")["variants"]["negro"]
        assert "almendra" in negro["extra_ingredients"]
        assert "ajonjoli" in negro["extra_ingredients"]

    def test_mole_negro_has_english_name(self):
        negro = retrieval.get_dish_data("mole")["variants"]["negro"]
        assert negro.get("name_en") == "Oaxacan Black Mole"

    def test_custom_returns_none(self):
        assert retrieval.get_dish_data("custom") is None

    def test_nonexistent_entity_returns_none(self):
        assert retrieval.get_dish_data("platillo_xyz_inexistente") is None


class TestYamlContextPreference:
    def test_mole_context_comes_from_yaml(self):
        ctx = retrieval.get_dish_context("mole")
        assert "Platillo: mole" in ctx

    def test_mole_context_includes_variantes_header(self):
        assert "Variantes" in retrieval.get_dish_context("mole")

    def test_mole_context_includes_english_name(self):
        ctx = retrieval.get_dish_context("mole")
        assert "Oaxacan" in ctx or "Black Mole" in ctx

    def test_mole_context_includes_base_ingredients(self):
        assert "chile_ancho" in retrieval.get_dish_context("mole")

    def test_arroz_falls_back_to_txt(self):
        # arroz has no .yaml yet — must fall back to .txt without error
        ctx = retrieval.get_dish_context("arroz")
        assert len(ctx) > 0


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
