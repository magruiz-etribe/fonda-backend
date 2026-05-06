from __future__ import annotations

import logging
import time
from typing import Any, Final, Literal

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

import config

logger = logging.getLogger(__name__)


Role = Literal["user", "agent"]


_client = boto3.client(
    "dynamodb",
    region_name=config.DDB_REGION,
    config=Config(retries={"max_attempts": 2, "mode": "standard"}),
)


_MAX_HISTORY_RETURN: Final[int] = 5


def get_history(session_id: str, limit: int = _MAX_HISTORY_RETURN) -> list[dict[str, str]]:
    if not session_id:
        return []
    try:
        resp = _client.get_item(
            TableName=config.DDB_TABLE_NAME,
            Key={"session_id": {"S": session_id}},
            ProjectionExpression="turns",
        )
    except (ClientError, BotoCoreError) as e:
        logger.warning(
            "ddb_get_history_failed",
            extra={"session_id": session_id, "error": str(e)},
        )
        return []

    item = resp.get("Item") or {}
    raw_turns = item.get("turns", {}).get("L", [])

    turns: list[dict[str, str]] = []
    for t in raw_turns:
        if not isinstance(t, dict):
            continue
        m = t.get("M") or {}
        role = m.get("role", {}).get("S")
        text = m.get("text", {}).get("S")
        if isinstance(role, str) and isinstance(text, str):
            turns.append({"role": role, "text": text})

    if limit > 0:
        return turns[-limit:]
    return turns


def append_turns(session_id: str, new_turns: list[dict[str, str]]) -> None:
    if not session_id or not new_turns:
        return

    items: list[dict[str, Any]] = []
    now = int(time.time())
    for t in new_turns:
        role = t.get("role")
        text = t.get("text")
        if not isinstance(role, str) or not isinstance(text, str) or not text:
            continue
        items.append({
            "M": {
                "role": {"S": role},
                "text": {"S": text},
                "ts": {"N": str(now)},
            }
        })

    if not items:
        return

    try:
        _client.update_item(
            TableName=config.DDB_TABLE_NAME,
            Key={"session_id": {"S": session_id}},
            UpdateExpression="SET turns = list_append(if_not_exists(turns, :empty), :new)",
            ExpressionAttributeValues={
                ":empty": {"L": []},
                ":new": {"L": items},
            },
        )
    except (ClientError, BotoCoreError) as e:
        logger.warning(
            "ddb_append_turns_failed",
            extra={"session_id": session_id, "error": str(e)},
        )
