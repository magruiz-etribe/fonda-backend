"""Tests for deterministic EXTRACTING fallback when the LLM returns empty questions."""
import json
import os

os.environ.setdefault("NOVA_2_LITE_MODEL_ID", "test-classifier-model")
os.environ.setdefault("NOVA_PRO_MODEL_ID", "test-generator-model")
os.environ.setdefault("DDB_TABLE_NAME", "test-table")

from unittest.mock import patch

import generation
import retrieval


_ENCHILADAS_KB = retrieval.get_dish_data("enchiladas") or {}
_BIRRIA_KB = retrieval.get_dish_data("birria") or {}


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
    @patch("bedrock_client.converse")
    def test_empty_llm_response_uses_deterministic_question(self, mock_converse):
        mock_converse.return_value = json.dumps({
            "response": [],
            "variables_complete": False,
            "collected_ingredients": ["pollo"],
            "buttons": [],
        })

        result = generation.generate_extracting(
            current_dish="enchiladas",
            companions=[],
            collected_ingredients=["pollo"],
            message="enchiladas de pollo",
            history=[],
            kb_data=_ENCHILADAS_KB,
        )

        assert mock_converse.call_count == 1
        assert result.variables_complete is False
        assert "salsa" in result.response[0].lower()
        assert result.collected_ingredients == ["pollo"]

    @patch("bedrock_client.converse")
    def test_empty_llm_with_all_vars_covered_transitions(self, mock_converse):
        mock_converse.return_value = json.dumps({
            "response": [],
            "variables_complete": False,
            "collected_ingredients": ["pollo", "verde"],
            "buttons": [],
        })

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
