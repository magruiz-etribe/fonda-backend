"""Unit tests for flags.py — pure Python, uses real KB reference files."""
import os

os.environ.setdefault("NOVA_2_LITE_MODEL_ID", "test-classifier-model")
os.environ.setdefault("NOVA_PRO_MODEL_ID", "test-generator-model")
os.environ.setdefault("DDB_TABLE_NAME", "test-table")

from flags import compute_flags


class TestAllergens:
    def test_almendra_triggers_nuts(self):
        assert "nuts" in compute_flags(["almendra"])["allergens"]

    def test_cacahuate_triggers_nuts(self):
        assert "nuts" in compute_flags(["cacahuate"])["allergens"]

    def test_ajonjoli_triggers_sesame(self):
        assert "sesame" in compute_flags(["ajonjoli"])["allergens"]

    def test_pan_triggers_gluten(self):
        flags = compute_flags(["pan"])
        assert "gluten" in flags["allergens"]
        assert flags["gluten_free"] is False

    def test_harina_de_trigo_triggers_gluten(self):
        assert "gluten" in compute_flags(["harina_de_trigo"])["allergens"]

    def test_huevo_triggers_eggs(self):
        assert "eggs" in compute_flags(["huevo"])["allergens"]

    def test_queso_triggers_dairy(self):
        assert "dairy" in compute_flags(["queso"])["allergens"]

    def test_camaron_triggers_shellfish(self):
        assert "shellfish" in compute_flags(["camaron"])["allergens"]

    def test_soya_triggers_soy(self):
        assert "soy" in compute_flags(["soya"])["allergens"]

    def test_multiple_allergens_detected(self):
        flags = compute_flags(["almendra", "pan", "huevo"])
        assert set(flags["allergens"]) >= {"nuts", "gluten", "eggs"}

    def test_no_allergen_triggers_empty_list(self):
        assert compute_flags(["jitomate", "cebolla", "ajo", "sal"])["allergens"] == []

    def test_allergens_list_is_sorted(self):
        flags = compute_flags(["pan", "almendra", "huevo"])
        assert flags["allergens"] == sorted(flags["allergens"])

    def test_extras_contribute_allergens(self):
        base = ["jitomate", "cebolla"]
        flags = compute_flags(base, extras=["almendra"])
        assert "nuts" in flags["allergens"]

    def test_case_insensitive_matching(self):
        flags = compute_flags(["ALMENDRA", "Pan"])
        assert "nuts" in flags["allergens"]
        assert "gluten" in flags["allergens"]


class TestGlutenFree:
    def test_gluten_free_when_no_trigger(self):
        assert compute_flags(["chile_ancho", "tomatillo", "cilantro"])["gluten_free"] is True

    def test_not_gluten_free_when_pan_present(self):
        assert compute_flags(["chile_ancho", "pan"])["gluten_free"] is False

    def test_gluten_free_with_nuts_but_no_gluten(self):
        flags = compute_flags(["almendra", "ajonjoli"])
        assert flags["gluten_free"] is True


class TestVegetarian:
    def test_no_meat_is_vegetarian(self):
        assert compute_flags(["tomatillo", "chile_serrano", "cilantro"])["vegetarian"] is True

    def test_caldo_de_pollo_is_not_vegetarian(self):
        assert compute_flags(["chile_ancho", "caldo_de_pollo"])["vegetarian"] is False

    def test_caldo_de_res_is_not_vegetarian(self):
        assert compute_flags(["fideo", "caldo_de_res"])["vegetarian"] is False

    def test_camaron_is_not_vegetarian(self):
        assert compute_flags(["chile_ancho", "camaron"])["vegetarian"] is False

    def test_queso_is_vegetarian(self):
        assert compute_flags(["queso", "tomatillo"])["vegetarian"] is True

    def test_huevo_is_vegetarian(self):
        assert compute_flags(["huevo", "jitomate"])["vegetarian"] is True


class TestVegan:
    def test_pure_vegetables_is_vegan(self):
        flags = compute_flags(["jitomate", "cebolla", "chile_serrano", "cilantro"])
        assert flags["vegan"] is True

    def test_caldo_de_pollo_is_not_vegan(self):
        assert compute_flags(["caldo_de_pollo", "jitomate"])["vegan"] is False

    def test_miel_is_not_vegan(self):
        assert compute_flags(["miel", "jitomate"])["vegan"] is False

    def test_mantequilla_is_not_vegan(self):
        assert compute_flags(["mantequilla"])["vegan"] is False

    def test_queso_is_not_vegan(self):
        assert compute_flags(["queso", "tomatillo"])["vegan"] is False

    def test_huevo_is_not_vegan(self):
        assert compute_flags(["huevo"])["vegan"] is False

    def test_vegetarian_not_implies_vegan(self):
        flags = compute_flags(["queso", "tomatillo"])
        assert flags["vegetarian"] is True
        assert flags["vegan"] is False


class TestSpicyLevel:
    def test_no_chiles_is_none(self):
        assert compute_flags(["jitomate", "cebolla", "ajo"])["spicy_level"] == "none"

    def test_chile_ancho_is_mild(self):
        assert compute_flags(["chile_ancho"])["spicy_level"] == "mild"

    def test_chile_guajillo_is_mild(self):
        assert compute_flags(["chile_guajillo"])["spicy_level"] == "mild"

    def test_chile_serrano_is_medium(self):
        assert compute_flags(["chile_serrano"])["spicy_level"] == "medium"

    def test_chile_chipotle_is_medium(self):
        assert compute_flags(["chile_chipotle"])["spicy_level"] == "medium"

    def test_chile_habanero_is_hot(self):
        assert compute_flags(["chile_habanero"])["spicy_level"] == "hot"

    def test_hot_beats_mild(self):
        assert compute_flags(["chile_ancho", "chile_habanero"])["spicy_level"] == "hot"

    def test_medium_beats_mild(self):
        assert compute_flags(["chile_ancho", "chile_serrano"])["spicy_level"] == "medium"

    def test_hot_beats_medium(self):
        assert compute_flags(["chile_serrano", "chile_habanero"])["spicy_level"] == "hot"


class TestEdgeCases:
    def test_empty_ingredients_returns_safe_defaults(self):
        flags = compute_flags([])
        assert flags["allergens"] == []
        assert flags["gluten_free"] is True
        assert flags["vegetarian"] is True
        assert flags["vegan"] is True
        assert flags["spicy_level"] == "none"

    def test_extras_none_behaves_like_empty(self):
        assert compute_flags(["jitomate"], None) == compute_flags(["jitomate"], [])

    def test_return_shape_always_complete(self):
        flags = compute_flags(["jitomate"])
        assert set(flags.keys()) == {"allergens", "gluten_free", "vegetarian", "vegan", "spicy_level"}

    def test_mole_negro_ingredients(self):
        """Integration smoke: base + negro variant ingredients produce expected flags."""
        base = [
            "chile_ancho", "chile_mulato", "chile_pasilla", "jitomate",
            "cebolla", "ajo", "chocolate_de_mesa", "caldo_de_pollo", "sal", "aceite",
        ]
        negro_extra = [
            "chile_chilhuacle_negro", "chile_mulato_oscuro", "chocolate_oaxaqueño",
            "tortilla_quemada", "pasitas", "almendra", "ajonjoli", "clavo", "canela",
        ]
        flags = compute_flags(base, negro_extra)
        assert "nuts" in flags["allergens"]
        assert "sesame" in flags["allergens"]
        assert "gluten" not in flags["allergens"]
        assert flags["gluten_free"] is True
        assert flags["vegetarian"] is False  # caldo_de_pollo
        assert flags["spicy_level"] == "mild"  # chile_ancho, chile_mulato, etc.

    def test_mole_poblano_has_gluten(self):
        """Mole poblano with pan should trigger gluten flag."""
        base = [
            "chile_ancho", "chile_mulato", "chile_pasilla", "jitomate",
            "cebolla", "ajo", "chocolate_de_mesa", "caldo_de_pollo", "sal", "aceite",
        ]
        poblano_extra = [
            "chile_chipotle", "almendra", "cacahuate", "ajonjoli",
            "pasitas", "canela", "clavo", "comino", "pan", "tortilla",
        ]
        flags = compute_flags(base, poblano_extra)
        assert "gluten" in flags["allergens"]
        assert flags["gluten_free"] is False
        assert "nuts" in flags["allergens"]
        assert "sesame" in flags["allergens"]
