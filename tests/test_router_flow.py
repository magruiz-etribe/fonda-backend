"""End-to-end router flow tests for the traduccion state machine."""
import os

os.environ.setdefault("NOVA_2_LITE_MODEL_ID", "test-classifier-model")
os.environ.setdefault("NOVA_PRO_MODEL_ID", "test-generator-model")
os.environ.setdefault("DDB_TABLE_NAME", "test-table")

from unittest.mock import patch

import router
from classifier import ClassifierResult
from generation import GenResult

_DRAFT_CARD = (
    "**Huevos Revueltos con Jamón**\n"
    "Huevos revueltos con jamón.\n\n"
    "**Scrambled Eggs with Ham**\n"
    "Scrambled eggs with ham.\n\n"
    "🎉 ¡Listo! Así quedó la descripción para tus clientes. ¿La guardamos en el menú?"
)


def _cr(dish: str = "huevos_revueltos", companions: list | None = None) -> ClassifierResult:
    return ClassifierResult(
        intent="traduccion",
        current_dish=dish,
        companions=companions or [],
        current_dishes=[dish] if dish else [],
    )


def _session(
    dish: str = "huevos_revueltos",
    status: str | None = None,
    collected: list | None = None,
) -> dict:
    return {
        "current_dish": dish,
        "companions": [],
        "dish_status": status,
        "collected_ingredients": collected or [],
        "detected_flags": [],
        "menu_del_dia": [],
    }


class TestHuevosConJamonFlow:
    @patch("router.cls_module.classify", return_value=_cr())
    @patch("flag_llm.compute_flags_for_dish")
    @patch("bedrock_client.converse_json")
    def test_huevo_con_jamon_skips_extracting_question(
        self, mock_json, mock_flags, _mock_cls,
    ):
        mock_flags.return_value = {
            "allergens": True,
            "allergen_triggers": ["huevo"],
            "gluten_free": True,
            "gluten_triggers": [],
            "vegetarian": False,
            "vegan": False,
            "spicy_level": "none",
            "spicy_triggers": [],
        }
        mock_json.return_value = {
            "response": [
                "**Huevos Revueltos con Jamón**\n"
                "Huevos revueltos con jamón.\n\n"
                "**Scrambled Eggs with Ham**\n"
                "Scrambled eggs with ham.\n\n"
                "🎉 ¡Listo!",
            ],
            "buttons": ["✅ Guardar en menú", "✏️ Hacer cambios"],
            "current_dishes": ["huevos_revueltos"],
        }

        result = router.handle(
            "huevo con jamon",
            _session(status=None),
            [],
        )

        assert result.dish_status == "DRAFTING"
        assert not any("jamón" in r.lower() and "?" in r for r in result.response)
        assert not any("lleva" in r.lower() for r in result.response)
        assert mock_json.call_count == 1
        stage = mock_json.call_args.kwargs.get("stage") or mock_json.call_args[1].get("stage")
        assert stage == "gen_drafting"

    @patch("router.cls_module.classify", return_value=_cr())
    @patch("flag_llm.compute_flags_for_dish")
    @patch("bedrock_client.converse_json")
    def test_save_button_works_on_first_click_in_drafting(
        self, mock_json, mock_flags, _mock_cls,
    ):
        mock_flags.return_value = {
            "allergens": True,
            "allergen_triggers": ["huevo"],
            "gluten_free": True,
            "gluten_triggers": [],
            "vegetarian": False,
            "vegan": False,
            "spicy_level": "none",
            "spicy_triggers": [],
        }
        history = [
            {"role": "user", "text": "huevo con jamon"},
            {"role": "agent", "text": _DRAFT_CARD},
        ]
        session = _session(status="DRAFTING", collected=["huevo", "jamon"])

        result = router.handle("✅ Guardar en menú", session, history)

        assert result.save_to_menu is True
        assert "guardé" in result.response[0].lower()
        assert result.dish_status is None
        mock_json.assert_not_called()

    @patch("router.cls_module.classify", return_value=_cr())
    @patch("flag_llm.compute_flags_for_dish")
    @patch("bedrock_client.converse_json")
    def test_save_works_even_if_session_status_is_confirming_flags(
        self, mock_json, mock_flags, _mock_cls,
    ):
        mock_flags.return_value = {
            "allergens": True,
            "allergen_triggers": ["huevo"],
            "gluten_free": True,
            "gluten_triggers": [],
            "vegetarian": False,
            "vegan": False,
            "spicy_level": "none",
            "spicy_triggers": [],
        }
        history = [
            {"role": "user", "text": "si"},
            {"role": "agent", "text": _DRAFT_CARD},
        ]
        session = _session(status="CONFIRMING_FLAGS", collected=["huevo", "jamon"])

        result = router.handle("✅ Guardar en menú", session, history)

        assert result.save_to_menu is True
        mock_json.assert_not_called()


class TestEnchiladasFlow:
    @patch("router.cls_module.classify", return_value=_cr("enchiladas"))
    @patch("bedrock_client.converse_json")
    def test_enchiladas_sin_relleno_pregunta_relleno(self, mock_json, _mock_cls):
        mock_json.return_value = {
            "response": ["¿Tu platillo lleva queso?"],
            "variables_complete": False,
            "collected_ingredients": [],
            "buttons": [],
        }

        result = router.handle("enchiladas", _session(dish="enchiladas"), [])

        assert result.dish_status == "EXTRACTING"
        assert result.variables_complete is False
        assert len(result.response) == 1
        assert "relleno" in result.response[0].lower() or "preparas" in result.response[0].lower()
        assert result.buttons

    @patch("router.cls_module.classify", return_value=_cr("enchiladas"))
    @patch("flag_llm.compute_flags_for_dish")
    @patch("bedrock_client.converse_json")
    def test_enchiladas_pollo_verde_short_circuits_to_draft(
        self, mock_json, mock_flags, _mock_cls,
    ):
        mock_flags.return_value = {
            "allergens": True,
            "allergen_triggers": ["queso"],
            "gluten_free": True,
            "gluten_triggers": [],
            "vegetarian": False,
            "vegan": False,
            "spicy_level": "none",
            "spicy_triggers": [],
        }
        mock_json.return_value = {
            "response": ["**Enchiladas**\nDesc.\n\n**Enchiladas**\nDesc.\n\n🎉 ¡Listo!"],
            "buttons": ["✅ Guardar en menú", "✏️ Hacer cambios"],
            "current_dishes": ["enchiladas"],
        }

        result = router.handle(
            "enchiladas verdes de pollo",
            _session(dish="enchiladas"),
            [],
        )

        assert result.dish_status == "DRAFTING"
        assert mock_json.call_count == 1
        assert mock_json.call_args.kwargs.get("stage") == "gen_drafting"


class TestMessageDetection:
    def test_is_save_request_matches_button_label(self):
        assert router._is_save_request("✅ Guardar en menú")

    def test_is_save_request_rejects_plain_si(self):
        assert not router._is_save_request("si")

    def test_is_affirmative_accepts_si(self):
        assert router._is_affirmative("si")

    def test_hidden_triggers_exclude_stated_jamon(self):
        kb = __import__("retrieval").get_dish_data("huevos_revueltos") or {}
        stated = router._user_stated_ingredients(
            ["huevo", "jamon"], "huevo con jamon", "huevos_revueltos", [], kb
        )
        hidden = router._filter_hidden_allergen_triggers(
            ["huevo", "jamon"], stated, kb
        )
        assert hidden == []
