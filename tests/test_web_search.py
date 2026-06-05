"""Tests for web_search.py and bedrock grounding response parsing."""

import os

os.environ.setdefault("NOVA_2_LITE_MODEL_ID", "test-classifier-model")
os.environ.setdefault("NOVA_PRO_MODEL_ID", "test-generator-model")
os.environ.setdefault("DDB_TABLE_NAME", "test-table")

from unittest.mock import patch

import bedrock_client
import web_search


def test_extract_grounding_log_data_parses_queries_and_citations():
    resp = {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "name": "nova_grounding",
                            "input": {"query": "mole poblano ingredientes"},
                        }
                    },
                    {
                        "text": "El mole poblano es una salsa tradicional...",
                        "citationsContent": {
                            "citations": [
                                {
                                    "location": {
                                        "web": {
                                            "url": "https://example.com/mole",
                                            "domain": "example.com",
                                        }
                                    }
                                }
                            ]
                        },
                    },
                ]
            }
        }
    }

    data = bedrock_client.extract_grounding_log_data(resp)

    assert data["queries"] == ["mole poblano ingredientes"]
    assert data["citations"] == [{"url": "https://example.com/mole", "domain": "example.com"}]
    assert "mole poblano" in data["text"]


def test_resolve_search_query_uses_current_message():
    q = web_search.resolve_search_query("enchiladas verdes", [], None)
    assert q == "enchiladas verdes"


def test_resolve_search_query_uses_colloquial_name():
    q = web_search.resolve_search_query("orejas de elefante", [], None)
    assert q == "orejas de elefante"


def test_resolve_search_query_falls_back_to_persisted_phrase():
    q = web_search.resolve_search_query(
        "listo",
        [{"role": "user", "text": "huevos con jamón"}],
        {"search_phrase": "huevos con jamón"},
    )
    assert q == "huevos con jamón"


def test_resolve_search_query_falls_back_to_history():
    q = web_search.resolve_search_query(
        "✅ Listo, eso es todo!",
        [{"role": "user", "text": "enchiladas verdes"}],
        {},
    )
    assert q == "enchiladas verdes"


@patch("web_search.config.WEB_GROUNDING_ENABLED", False)
def test_search_platillo_skipped_when_disabled():
    assert web_search.search_platillo("mole") is None


@patch("web_search.config.WEB_GROUNDING_ENABLED", True)
def test_search_platillo_allows_custom_colloquial_names():
    with patch("web_search.bedrock_client.converse") as mock_converse:
        mock_converse.return_value = {
            "output": {"message": {"content": [{"text": "Milanesa empanizada."}]}}
        }
        result = web_search.search_platillo("orejas de elefante")
    assert result is not None


@patch("web_search.config.WEB_GROUNDING_ENABLED", True)
@patch("web_search.bedrock_client.converse")
def test_search_platillo_prompt_uses_client_phrase(mock_converse):
    mock_converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {"text": "Ingredientes: totopos.\nVariantes: verdes."},
                ]
            }
        }
    }

    web_search.search_platillo("enchiladas verdes")

    call_args = mock_converse.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[2]
    user_text = messages[0]["content"][0]["text"]
    assert "enchiladas verdes" in user_text
    assert "comida mexicana" in user_text.lower()
    assert "coloquial" in user_text.lower()


def test_converse_return_full_without_text_blocks():
    """Grounding responses may have only toolUse blocks — must not fail."""
    from unittest.mock import patch

    resp = {
        "stopReason": "end_turn",
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "name": "nova_grounding",
                            "input": {"query": "mole poblano"},
                        }
                    }
                ]
            }
        },
    }
    with patch.object(bedrock_client, "_client") as mock_client:
        mock_client.converse.return_value = resp
        result = bedrock_client.converse(
            "test-model",
            "",
            [{"role": "user", "content": [{"text": "test"}]}],
            tool_config={"tools": [{"systemTool": {"name": "nova_grounding"}}]},
            return_full=True,
        )
    assert result == resp
