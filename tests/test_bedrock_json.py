"""Tests for lenient JSON parsing of LLM responses."""
import json
import os

os.environ.setdefault("NOVA_2_LITE_MODEL_ID", "test-classifier-model")
os.environ.setdefault("NOVA_PRO_MODEL_ID", "test-generator-model")
os.environ.setdefault("DDB_TABLE_NAME", "test-table")

import bedrock_client
import generation


class TestParseJsonLenient:
    def test_parses_json_with_literal_newlines_in_strings(self):
        raw = """{
  "response": ["**Enchiladas**
Tortilla rellena de pollo.

**Chicken Enchiladas**
Corn tortilla stuffed with chicken.", "🎉 ¡Listo!"],
  "buttons": ["✅ Guardar en menú"]
}"""
        data = bedrock_client.parse_json_lenient(raw)
        assert isinstance(data, dict)
        assert len(data["response"]) == 2
        assert "**Enchiladas**" in data["response"][0]
        assert "Chicken Enchiladas" in data["response"][0]

    def test_parses_json_inside_markdown_fence(self):
        raw = """```json
{"response": ["hola"], "buttons": []}
```"""
        data = bedrock_client.parse_json_lenient(raw)
        assert data["response"] == ["hola"]

    def test_parses_json_with_trailing_comma(self):
        raw = '{"response": ["hola",], "buttons": [],}'
        data = bedrock_client.parse_json_lenient(raw)
        assert data["response"] == ["hola"]


class TestDraftingParse:
    def test_try_parse_drafting_accepts_lenient_json(self):
        raw = {
            "response": [
                "**Mole Negro**\nSalsa intensa de chiles.\n\n**Black Mole**\nRich chili sauce.",
                "🎉 ¡Listo!",
            ],
            "buttons": [],
            "current_dishes": ["mole"],
        }
        result = generation._try_parse_drafting_data(raw, "mole", ["arroz"])
        assert result is not None
        assert len(result.response) == 2
        assert result.buttons == ["✅ Guardar en menú", "✏️ Hacer cambios"]
        assert result.current_dishes == ["mole", "arroz"]

    def test_try_parse_drafting_rejects_invalid_payload(self):
        assert generation._try_parse_drafting_data("not json at all", "mole", []) is None


class TestStructuredToolConfig:
    def test_extract_tool_input_from_response(self):
        resp = {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "draft_menu_card",
                                "input": {
                                    "response": ["hola"],
                                    "buttons": [],
                                    "current_dishes": ["mole"],
                                },
                            }
                        }
                    ]
                }
            }
        }
        data = bedrock_client._extract_tool_input(resp, "draft_menu_card")
        assert data["response"] == ["hola"]

    def test_extract_tool_input_rejects_empty_dict(self):
        resp = {
            "output": {
                "message": {
                    "content": [
                        {"toolUse": {"name": "draft_menu_card", "input": {}}},
                    ]
                }
            }
        }
        try:
            bedrock_client._extract_tool_input(resp, "draft_menu_card")
            assert False, "expected BedrockError"
        except bedrock_client.BedrockError:
            pass
