"""Validate Bedrock JSON schemas and structured-output wiring."""
import json
import os
from unittest.mock import patch

os.environ.setdefault("NOVA_2_LITE_MODEL_ID", "test-classifier-model")
os.environ.setdefault("NOVA_PRO_MODEL_ID", "test-generator-model")
os.environ.setdefault("DDB_TABLE_NAME", "test-table")

import bedrock_client
import llm_schemas


class TestBedrockSchemas:
    def test_all_registered_schemas_are_valid(self):
        llm_schemas.assert_all_schemas_valid()

    def test_schemas_are_json_serializable(self):
        for name, schema in llm_schemas.ALL_SCHEMAS.items():
            serialized = json.dumps(schema)
            assert serialized.startswith("{"), name

    def test_generation_includes_optional_confirmation_fields(self):
        props = llm_schemas.GENERATION["properties"]
        for field in (
            "completeness_confirmed",
            "allergens_confirmed",
            "gluten_confirmed",
            "spicy_confirmed",
        ):
            assert field in props, field
            assert field not in llm_schemas.GENERATION["required"]

    def test_structured_tool_config_uses_strict_by_default(self):
        cfg = bedrock_client.structured_tool_config("draft_menu_card", llm_schemas.DRAFTING)
        tool_spec = cfg["tools"][0]["toolSpec"]
        assert tool_spec["strict"] is True
        assert cfg["toolChoice"] == {"tool": {"name": "draft_menu_card"}}


class TestConverseJsonFallback:
    @patch("bedrock_client._invoke")
    def test_parses_json_text_from_structured_response(self, mock_invoke):
        structured_resp = {
            "stopReason": "end_turn",
            "output": {"message": {"content": [{"text": '{"response": ["hola"], "buttons": []}'}]}},
        }
        mock_invoke.return_value = structured_resp

        data = bedrock_client.converse_json(
            "test-model",
            "system",
            [{"role": "user", "content": [{"text": "hi"}]}],
            schema=llm_schemas.CONFIRMING_FLAGS,
            tool_name="confirm_flags",
        )
        assert data["response"] == ["hola"]
        assert mock_invoke.call_count == 1

    @patch("bedrock_client._invoke")
    def test_falls_back_to_plain_converse_when_structured_response_unusable(self, mock_invoke):
        structured_resp = {
            "stopReason": "end_turn",
            "output": {"message": {"content": [{"text": "no json here"}]}},
        }
        mock_invoke.side_effect = [structured_resp, structured_resp, '{"response": ["fallback"], "buttons": []}']

        data = bedrock_client.converse_json(
            "test-model",
            "system",
            [{"role": "user", "content": [{"text": "hi"}]}],
            schema=llm_schemas.CONFIRMING_FLAGS,
            tool_name="confirm_flags",
        )
        assert data["response"] == ["fallback"]
        assert mock_invoke.call_count == 3
        assert mock_invoke.call_args_list[-1].args[4] is None

    @patch("bedrock_client._invoke")
    def test_parses_tool_use_input_when_present(self, mock_invoke):
        structured_resp = {
            "stopReason": "tool_use",
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "extract_variables",
                                "input": {
                                    "response": [],
                                    "variables_complete": True,
                                    "collected_ingredients": ["huevo"],
                                    "buttons": [],
                                },
                            }
                        }
                    ]
                }
            },
        }
        mock_invoke.return_value = structured_resp

        data = bedrock_client.converse_json(
            "test-model",
            "system",
            [{"role": "user", "content": [{"text": "hi"}]}],
            schema=llm_schemas.EXTRACTING,
            tool_name="extract_variables",
        )
        assert data["variables_complete"] is True
        assert mock_invoke.call_count == 1

    @patch("bedrock_client._invoke")
    def test_coerces_string_tool_input(self, mock_invoke):
        structured_resp = {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "compute_flags",
                                "input": '{"reasoning":"ok","allergens":false,"allergen_triggers":[],"gluten_free":true,"gluten_triggers":[],"vegetarian":true,"vegan":false,"spicy_level":"none","spicy_triggers":[]}',
                            }
                        }
                    ]
                }
            }
        }
        mock_invoke.return_value = structured_resp

        data = bedrock_client.converse_json(
            "test-model",
            "system",
            [{"role": "user", "content": [{"text": "hi"}]}],
            schema=llm_schemas.FLAGS,
            tool_name="compute_flags",
        )
        assert data["spicy_level"] == "none"
