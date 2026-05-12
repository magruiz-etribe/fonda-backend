"""
Flow tests for router.handle() — full pipeline with mocked Bedrock.

Each test mocks bedrock_client.converse with a (classifier_response, generation_response)
pair, then asserts the GenResult matches what the frontend would receive.

Classifier JSON shape:  {reasoning, intent, current_dishes, translate_now, pending_variant_for}
Generation JSON shape:  {response[], current_dishes[], buttons[]}
"""
import json
import os

os.environ.setdefault("NOVA_LITE_MODEL_ID", "test-model")
os.environ.setdefault("DDB_TABLE_NAME", "test-table")

from unittest.mock import patch

import pytest

import router
from generation import GenResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cls(
    intent="traduccion",
    current_dishes=None,
    translate_now=False,
    pending_variant_for=None,
) -> str:
    return json.dumps({
        "reasoning": "test",
        "intent": intent,
        "current_dishes": current_dishes or [],
        "translate_now": translate_now,
        "pending_variant_for": pending_variant_for,
    })


def _gen(response=None, current_dishes=None, buttons=None) -> str:
    return json.dumps({
        "response": response or ["respuesta"],
        "current_dishes": current_dishes or [],
        "buttons": buttons or [],
    })


def _handle(converse_mock, message, current_dishes, history, cls_kwargs, gen_kwargs):
    converse_mock.side_effect = [_cls(**cls_kwargs), _gen(**gen_kwargs)]
    return router.handle(message, current_dishes, history)


# ---------------------------------------------------------------------------
# Scenario 1: mole con arroz — step by step
# ---------------------------------------------------------------------------

class TestMoleArrozFlow:
    """
    Full flow: user says 'mole con arroz'.
    Expected turn sequence:
      1. Agent asks mole variant (buttons, 1 bubble)
      2. Agent asks arroz variant (buttons, 1 bubble)
      3. Agent shows ONE Spanish description card (2 bubbles + Traducir button)
      4. Agent shows ONE English translation (2 bubbles, no buttons, current_dishes=[])
    """

    @patch("bedrock_client.converse")
    def test_turn1_asks_mole_variant_only(self, mock_cv):
        result = _handle(
            mock_cv,
            message="es mole con arroz",
            current_dishes=[],
            history=[],
            cls_kwargs=dict(
                intent="traduccion",
                current_dishes=["mole", "arroz"],
                pending_variant_for="mole",
            ),
            gen_kwargs=dict(
                response=["¿Qué tipo de mole preparas? 🌶️ Te dejo algunas opciones 👇"],
                current_dishes=["mole", "arroz"],
                buttons=["🔴 Rojo", "⚫ Negro", "🌿 Verde", "🫑 Poblano"],
            ),
        )
        # One bubble only — no ingredient question mixed in
        assert len(result.response) == 1
        # Buttons present for mole variant selection
        assert len(result.buttons) >= 2
        # Dishes preserved
        assert result.current_dishes == ["mole", "arroz"]
        # Should NOT ask about arroz in the same bubble
        assert "arroz" not in result.response[0].lower()

    @patch("bedrock_client.converse")
    def test_turn2_asks_arroz_variant_only(self, mock_cv):
        history = [
            {"role": "user", "text": "es mole con arroz"},
            {"role": "agent", "text": "¿Qué tipo de mole preparas? 🌶️"},
        ]
        result = _handle(
            mock_cv,
            message="⚫ Negro",
            current_dishes=["mole", "arroz"],
            history=history,
            cls_kwargs=dict(
                intent="traduccion",
                current_dishes=["mole", "arroz"],
                pending_variant_for="arroz",
            ),
            gen_kwargs=dict(
                response=["¿Qué tipo de arroz preparas? 🍚 Te dejo algunas opciones 👇"],
                current_dishes=["mole", "arroz"],
                buttons=["🟥 Rojo", "⬜ Blanco", "🌿 Verde"],
            ),
        )
        # One bubble only
        assert len(result.response) == 1
        # Buttons present for arroz
        assert len(result.buttons) >= 2
        assert result.current_dishes == ["mole", "arroz"]

    @patch("bedrock_client.converse")
    def test_turn3_single_spanish_description_card(self, mock_cv):
        """After both variants confirmed, get ONE card for mole with arroz as side."""
        history = [
            {"role": "user", "text": "es mole con arroz"},
            {"role": "agent", "text": "¿Qué tipo de mole preparas?"},
            {"role": "user", "text": "⚫ Negro"},
            {"role": "agent", "text": "¿Qué tipo de arroz preparas?"},
        ]
        result = _handle(
            mock_cv,
            message="🟥 Rojo",
            current_dishes=["mole", "arroz"],
            history=history,
            cls_kwargs=dict(
                intent="traduccion",
                current_dishes=["mole", "arroz"],
                pending_variant_for=None,
            ),
            gen_kwargs=dict(
                response=[
                    "**Mole Negro** 🌑\nIntensa salsa de chiles negros y chocolate, servida con arroz rojo.",
                    "¿Te parece bien? 😊 Si quieres cambiar algo solo dime, o si ya está listo presiona el botón 👇",
                ],
                current_dishes=["mole", "arroz"],
                buttons=["✅ Traducir"],
            ),
        )
        # Two bubbles: description card + confirmation question
        assert len(result.response) == 2
        assert result.buttons == ["✅ Traducir"]
        # First bubble is the description card — main dish as title
        assert "**Mole Negro**" in result.response[0]
        # Arroz mentioned AS SIDE in the description, not as a separate card
        assert "arroz" in result.response[0].lower()
        # No separate Arroz card in response
        assert "**Arroz" not in result.response[0]
        assert "**Arroz" not in result.response[1]
        # Dishes preserved until translation
        assert result.current_dishes == ["mole", "arroz"]

    @patch("bedrock_client.converse")
    def test_turn4_translation_clears_dishes(self, mock_cv):
        """translate_now=true → ETAPA C → current_dishes cleared, no buttons."""
        history = [
            {"role": "user", "text": "🟥 Rojo"},
            {"role": "agent", "text": "**Mole Negro** 🌑\n...\n\n¿Te parece bien?"},
        ]
        result = _handle(
            mock_cv,
            message="✅ Traducir",
            current_dishes=["mole", "arroz"],
            history=history,
            cls_kwargs=dict(
                intent="traduccion",
                current_dishes=["mole", "arroz"],
                translate_now=True,
                pending_variant_for=None,
            ),
            gen_kwargs=dict(
                response=[
                    "**Black Mole**\nDark, complex sauce of toasted chiles and chocolate, served with red rice.",
                    "¡Tu traducción está lista! 🎉 Si quieres ajustar algo solo dime, o si tienes otro platillo aquí estoy 😊",
                ],
                current_dishes=[],
                buttons=[],
            ),
        )
        # Two bubbles: English card + closing message
        assert len(result.response) == 2
        # No buttons after translation
        assert result.buttons == []
        # current_dishes cleared
        assert result.current_dishes == []
        # English card is first bubble
        assert "**Black Mole**" in result.response[0]

    @patch("bedrock_client.converse")
    def test_turn5_post_translation_closure(self, mock_cv):
        """After translation delivered, user says 'gracias' → 1 short bubble."""
        history = [
            {"role": "agent", "text": "**Black Mole**\n...\n\n¡Tu traducción está lista! 🎉"},
        ]
        result = _handle(
            mock_cv,
            message="gracias",
            current_dishes=[],
            history=history,
            cls_kwargs=dict(
                intent="traduccion",
                current_dishes=[],
                pending_variant_for=None,
            ),
            gen_kwargs=dict(
                response=["¡De nada! 😊 Si necesitas traducir otro platillo o ayuda con algo más, aquí estoy."],
                current_dishes=[],
                buttons=[],
            ),
        )
        assert len(result.response) == 1
        assert result.buttons == []
        assert result.current_dishes == []


# ---------------------------------------------------------------------------
# Scenario 2: single KB dish (mole only)
# ---------------------------------------------------------------------------

class TestSingleDishFlow:
    @patch("bedrock_client.converse")
    def test_mole_asks_variant_with_buttons(self, mock_cv):
        result = _handle(
            mock_cv,
            message="quiero traducir mi mole",
            current_dishes=[],
            history=[],
            cls_kwargs=dict(
                intent="traduccion",
                current_dishes=["mole"],
                pending_variant_for="mole",
            ),
            gen_kwargs=dict(
                response=["¡Qué rico! 🌶️ ¿Qué tipo de mole preparas? 👇"],
                current_dishes=["mole"],
                buttons=["🔴 Rojo", "⚫ Negro", "🌿 Verde", "🫑 Poblano"],
            ),
        )
        assert len(result.response) == 1
        assert len(result.buttons) >= 2
        assert result.current_dishes == ["mole"]

    @patch("bedrock_client.converse")
    def test_custom_dish_no_buttons_asks_ingredients(self, mock_cv):
        result = _handle(
            mock_cv,
            message="quiero traducir mi enchilada especial",
            current_dishes=[],
            history=[],
            cls_kwargs=dict(
                intent="traduccion",
                current_dishes=["custom"],
                pending_variant_for=None,
            ),
            gen_kwargs=dict(
                response=["¡Me encanta! 🍳 ¿Me puedes platicar cuáles son los ingredientes principales?"],
                current_dishes=["custom"],
                buttons=[],
            ),
        )
        assert len(result.response) == 1
        assert result.buttons == []
        assert result.current_dishes == ["custom"]


# ---------------------------------------------------------------------------
# Scenario 3: non-translation intents
# ---------------------------------------------------------------------------

class TestNonTranslationIntents:
    @patch("bedrock_client.converse")
    def test_maps_intent(self, mock_cv):
        result = _handle(
            mock_cv,
            message="cómo me registro en Google Maps?",
            current_dishes=[],
            history=[],
            cls_kwargs=dict(intent="maps"),
            gen_kwargs=dict(
                response=["Para registrarte en Google Maps, sigue estos pasos 📍"],
                current_dishes=[],
                buttons=[],
            ),
        )
        assert result.current_dishes == []
        assert result.buttons == []

    @patch("bedrock_client.converse")
    def test_higiene_intent(self, mock_cv):
        result = _handle(
            mock_cv,
            message="qué debo hacer con los alimentos crudos?",
            current_dishes=[],
            history=[],
            cls_kwargs=dict(intent="higiene"),
            gen_kwargs=dict(
                response=["Separa siempre los alimentos crudos de los cocidos 🧼"],
                current_dishes=[],
                buttons=[],
            ),
        )
        assert result.current_dishes == []

    @patch("bedrock_client.converse")
    def test_fallback_intent(self, mock_cv):
        result = _handle(
            mock_cv,
            message="cuál es el precio del dólar?",
            current_dishes=[],
            history=[],
            cls_kwargs=dict(intent="fallback"),
            gen_kwargs=dict(
                response=["Lo siento, solo puedo ayudarte con traducciones, Google Maps e higiene 😊"],
                current_dishes=[],
                buttons=[],
            ),
        )
        assert result.current_dishes == []
        assert result.buttons == []


# ---------------------------------------------------------------------------
# Scenario 4: error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @patch("bedrock_client.converse")
    def test_classifier_bedrock_error_still_responds(self, mock_cv):
        from bedrock_client import BedrockError
        # Classifier fails; generation also fails → router returns fallback
        mock_cv.side_effect = BedrockError("timeout")
        result = router.handle("hola", [], [])
        # Should return some response, not raise
        assert isinstance(result, GenResult)
        assert len(result.response) >= 1
        assert result.current_dishes == []

    @patch("bedrock_client.converse")
    def test_malformed_classifier_json_uses_fallback_intent(self, mock_cv):
        """Classifier returns invalid JSON → classifier falls back → generation runs."""
        mock_cv.side_effect = [
            "this is not json at all",  # classifier broken
            _gen(
                response=["Disculpa, tuve un problema. ¿Puedes repetir? 😊"],
                current_dishes=[],
                buttons=[],
            ),
        ]
        result = router.handle("hola", [], [])
        assert len(result.response) >= 1

    @patch("bedrock_client.converse")
    def test_malformed_generation_json_uses_fallback_response(self, mock_cv):
        """Generation returns invalid JSON → GenResult fallback."""
        mock_cv.side_effect = [
            _cls(intent="traduccion", current_dishes=["mole"]),
            "not valid json",  # generation broken
        ]
        result = router.handle("mole", ["mole"], [])
        assert len(result.response) >= 1


# ---------------------------------------------------------------------------
# Scenario 5: classifier pending_variant_for parsing
# ---------------------------------------------------------------------------

class TestClassifierParsing:
    """Unit tests for ClassifierResult parsing — no router needed."""

    def test_pending_variant_for_extracted(self):
        from classifier import _parse
        raw = json.dumps({
            "reasoning": "mole needs variant",
            "intent": "traduccion",
            "current_dishes": ["mole", "arroz"],
            "translate_now": False,
            "pending_variant_for": "mole",
        })
        cr = _parse(raw, [])
        assert cr.pending_variant_for == "mole"

    def test_pending_variant_for_null(self):
        from classifier import _parse
        raw = json.dumps({
            "reasoning": "all confirmed",
            "intent": "traduccion",
            "current_dishes": ["mole"],
            "translate_now": False,
            "pending_variant_for": None,
        })
        cr = _parse(raw, [])
        assert cr.pending_variant_for is None

    def test_pending_variant_for_string_null_treated_as_none(self):
        from classifier import _parse
        raw = json.dumps({
            "reasoning": "all confirmed",
            "intent": "traduccion",
            "current_dishes": ["mole"],
            "translate_now": False,
            "pending_variant_for": "null",
        })
        cr = _parse(raw, [])
        assert cr.pending_variant_for is None

    def test_non_traduccion_intent_clears_pending(self):
        from classifier import _parse
        raw = json.dumps({
            "reasoning": "maps",
            "intent": "maps",
            "current_dishes": [],
            "translate_now": False,
            "pending_variant_for": None,
        })
        cr = _parse(raw, ["mole"])
        assert cr.pending_variant_for is None
        assert cr.current_dishes == []

    def test_translate_now_parsed(self):
        from classifier import _parse
        raw = json.dumps({
            "reasoning": "user clicked Traducir",
            "intent": "traduccion",
            "current_dishes": ["mole"],
            "translate_now": True,
            "pending_variant_for": None,
        })
        cr = _parse(raw, [])
        assert cr.translate_now is True
