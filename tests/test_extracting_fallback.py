"""Tests for deterministic EXTRACTING fallback when the LLM returns empty questions."""
import json
import os

os.environ.setdefault("NOVA_2_LITE_MODEL_ID", "test-classifier-model")
os.environ.setdefault("NOVA_PRO_MODEL_ID", "test-generator-model")
os.environ.setdefault("DDB_TABLE_NAME", "test-table")

from unittest.mock import patch

import generation
import retrieval

retrieval.get_dish_data.cache_clear()

_ENCHILADAS_KB = retrieval.get_dish_data("enchiladas") or {}
_BIRRIA_KB = retrieval.get_dish_data("birria") or {}
_HUEVOS_KB = retrieval.get_dish_data("huevos_revueltos") or {}


class TestPrefillCollected:
    def test_message_huevo_con_jamon_prefills_jamon(self):
        prefilled = generation._prefill_collected_from_message(
            "huevo con jamon", [], _HUEVOS_KB
        )
        assert "jamon" in prefilled


class TestExtractingDeterministicFallback:
    def test_builds_question_for_first_missing_variable(self):
        result = generation._build_extracting_deterministic_fallback(
            current_dish="enchiladas",
            collected_ingredients=[],
            kb_data=_ENCHILADAS_KB,
        )
        assert result is not None
        assert result.variables_complete is False
        assert len(result.response) == 1
        assert "relleno" in result.response[0].lower()
        assert result.buttons == ["Pollo", "Queso", "Res", "Frijoles"]

    def test_skips_covered_variable_and_asks_next(self):
        result = generation._build_extracting_deterministic_fallback(
            current_dish="enchiladas",
            collected_ingredients=["pollo"],
            kb_data=_ENCHILADAS_KB,
        )
        assert result is not None
        assert result.variables_complete is False
        assert "salsa" in result.response[0].lower()
        assert result.buttons == ["Verde", "Roja", "Mole", "Suizas (crema)"]
        assert result.collected_ingredients == ["pollo"]

    def test_marks_complete_when_all_variables_covered(self):
        result = generation._build_extracting_deterministic_fallback(
            current_dish="enchiladas",
            collected_ingredients=["pollo", "salsa verde"],
            kb_data=_ENCHILADAS_KB,
        )
        assert result is not None
        assert result.variables_complete is True
        assert result.response == []

    def test_birria_protein_question(self):
        result = generation._build_extracting_deterministic_fallback(
            current_dish="birria",
            collected_ingredients=[],
            kb_data=_BIRRIA_KB,
        )
        assert result is not None
        assert "proteína" in result.response[0].lower()
        assert "Res" in result.buttons


class TestGenerateExtractingUsesFallback:
    @patch("bedrock_client.converse_json")
    def test_empty_llm_response_uses_deterministic_question(self, mock_converse_json):
        mock_converse_json.return_value = {
            "response": [],
            "variables_complete": False,
            "collected_ingredients": ["pollo"],
            "buttons": [],
        }

        result = generation.generate_extracting(
            current_dish="enchiladas",
            companions=[],
            collected_ingredients=["pollo"],
            message="enchiladas de pollo",
            history=[],
            kb_data=_ENCHILADAS_KB,
        )

        assert mock_converse_json.call_count == 1
        assert result.variables_complete is False
        assert "salsa" in result.response[0].lower()
        assert result.collected_ingredients == ["pollo"]

    @patch("bedrock_client.converse_json")
    def test_empty_llm_with_all_vars_covered_transitions(self, mock_converse_json):
        mock_converse_json.return_value = {
            "response": [],
            "variables_complete": False,
            "collected_ingredients": ["pollo", "verde"],
            "buttons": [],
        }

        result = generation.generate_extracting(
            current_dish="enchiladas",
            companions=[],
            collected_ingredients=["pollo", "verde"],
            message="enchiladas verdes de pollo",
            history=[],
            kb_data=_ENCHILADAS_KB,
        )

        assert result.variables_complete is True
        assert result.response == []

    @patch("bedrock_client.converse_json")
    def test_injects_buttons_when_llm_omits_them(self, mock_converse_json):
        mock_converse_json.return_value = {
            "response": ["¿Con qué relleno preparas enchiladas?"],
            "variables_complete": False,
            "collected_ingredients": [],
            "buttons": [],
        }

        result = generation.generate_extracting(
            current_dish="enchiladas",
            companions=[],
            collected_ingredients=[],
            message="enchiladas",
            history=[],
            kb_data=_ENCHILADAS_KB,
        )

        assert result.buttons == ["Pollo", "Queso", "Res", "Frijoles"]

    @patch("bedrock_client.converse_json")
    def test_huevos_con_jamon_skips_redundant_questions(self, mock_converse_json):
        mock_converse_json.return_value = {
            "response": [
                "¿Con qué preparas los huevos revueltos?",
                "¿Qué más le pones a los huevos revueltos?",
            ],
            "variables_complete": False,
            "collected_ingredients": ["huevo", "jamón"],
            "buttons": [],
        }

        result = generation.generate_extracting(
            current_dish="huevos_revueltos",
            companions=[],
            collected_ingredients=["huevo", "jamón"],
            message="huevos con jamón",
            history=[],
            kb_data=_HUEVOS_KB,
        )

        assert result.variables_complete is True
        assert result.response == []
        assert result.collected_ingredients == ["huevo", "jamon"]

    @patch("bedrock_client.converse_json")
    def test_huevos_sin_acompanamiento_asks_once_with_buttons(self, mock_converse_json):
        mock_converse_json.return_value = {
            "response": ["¿Con qué acompañas los huevos?"],
            "variables_complete": False,
            "collected_ingredients": ["huevo"],
            "buttons": [],
        }

        result = generation.generate_extracting(
            current_dish="huevos_revueltos",
            companions=[],
            collected_ingredients=["huevo"],
            message="huevos revueltos",
            history=[],
            kb_data=_HUEVOS_KB,
        )

        assert result.variables_complete is False
        assert len(result.response) == 1
        assert "acompañ" in result.response[0].lower()
        assert "Jamón" in result.buttons

    @patch("bedrock_client.converse_json")
    def test_huevos_con_jamon_in_message_even_if_llm_misses_jamon(self, mock_converse_json):
        mock_converse_json.return_value = {
            "response": ["¿Con qué acompañas los huevos revueltos?"],
            "variables_complete": False,
            "collected_ingredients": ["huevo"],
            "buttons": [],
        }

        result = generation.generate_extracting(
            current_dish="huevos_revueltos",
            companions=[],
            collected_ingredients=[],
            message="huevo con jamon",
            history=[],
            kb_data=_HUEVOS_KB,
        )

        assert result.variables_complete is True
        assert result.response == []
        assert "jamon" in result.collected_ingredients
