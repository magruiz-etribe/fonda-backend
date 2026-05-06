from __future__ import annotations

import os
from typing import Final


def _required(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"missing required env var: {name}")
    return val


NOVA_MICRO_MODEL_ID: Final[str] = _required("NOVA_MICRO_MODEL_ID")
NOVA_LITE_MODEL_ID: Final[str] = _required("NOVA_LITE_MODEL_ID")
NOVA_PRO_MODEL_ID: Final[str] = _required("NOVA_PRO_MODEL_ID")

AWS_REGION: Final[str] = os.environ.get("AWS_REGION", "us-east-1")

DDB_TABLE_NAME: Final[str] = _required("DDB_TABLE_NAME")
DDB_REGION: Final[str] = os.environ.get("DDB_REGION", "us-east-1")

_DEFAULT_KB_PATH: Final[str] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kb"
)
KB_PATH: Final[str] = os.environ.get("KB_PATH", _DEFAULT_KB_PATH)

MAX_HISTORY_TURNS: Final[int] = 5
MAX_REPLY_LEN: Final[int] = 4000

CLASSIFIER_MAX_TOKENS: Final[int] = 200
ENTITY_MAPPER_MAX_TOKENS: Final[int] = 250
GEN_MAX_TOKENS: Final[int] = 800
IMAGE_MAX_TOKENS: Final[int] = 1000

BEDROCK_READ_TIMEOUT_S: Final[int] = 60
BEDROCK_CONNECT_TIMEOUT_S: Final[int] = 10

GENERIC_FALLBACK_REPLY: Final[str] = (
    "Disculpa, tuve un problema procesando tu mensaje. "
    "¿Puedes intentarlo de nuevo en un momento?"
)
