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


@patch("web_search.config.WEB_GROUNDING_ENABLED", False)
def test_search_platillo_skipped_when_disabled():
    assert web_search.search_platillo("mole") is None


@patch("web_search.config.WEB_GROUNDING_ENABLED", True)
@patch("web_search.bedrock_client.converse")
def test_search_platillo_logs_when_enabled(mock_converse):
    mock_converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "name": "nova_grounding",
                            "input": {"query": "chilaquiles rojos receta"},
                        }
                    },
                    {"text": "Los chilaquiles son totopos bañados en salsa."},
                ]
            }
        }
    }

    result = web_search.search_platillo("chilaquiles")

    assert result is not None
    assert result["queries"] == ["chilaquiles rojos receta"]
    assert "chilaquiles" in result["text"]
    mock_converse.assert_called_once()
    call_kwargs = mock_converse.call_args.kwargs
    assert call_kwargs["tool_config"] == web_search._GROUNDING_TOOL
    assert call_kwargs["return_full"] is True
