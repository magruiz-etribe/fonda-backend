from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Final

import config
import history_store
import router
import timing

logger = logging.getLogger()
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

_MAX_BODY_BYTES: Final[int] = 10 * 1024 * 1024

_CORS_ALLOW_ORIGIN: Final[str] = os.environ.get("CORS_ALLOW_ORIGIN", "*")
_CORS_ALLOW_HEADERS: Final[str] = os.environ.get(
    "CORS_ALLOW_HEADERS",
    "Content-Type,Authorization,X-Api-Key,X-Amz-Date,X-Amz-Security-Token",
)
_CORS_ALLOW_METHODS: Final[str] = "POST,OPTIONS"
_CORS_HEADERS: Final[dict[str, str]] = {
    "Access-Control-Allow-Origin": _CORS_ALLOW_ORIGIN,
    "Access-Control-Allow-Methods": _CORS_ALLOW_METHODS,
    "Access-Control-Allow-Headers": _CORS_ALLOW_HEADERS,
}


class _BadRequest(Exception):
    pass


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    request_id = getattr(context, "aws_request_id", "unknown")

    if _is_preflight(event):
        return _response(204, None)

    try:
        body = _parse_body(event)
        session_id = _require_str(body, "session_id")
        message = _require_str(body, "message")
    except _BadRequest as e:
        logger.warning("bad_request", extra={"request_id": request_id, "error": str(e)})
        return _response(400, {"error": str(e)})

    if len(message) > 250:
        logger.warning("message_too_long", extra={"request_id": request_id, "msg_len": len(message)})
        return _response(400, {"error": "Mensaje demasiado largo. Máximo 250 caracteres."})

    logger.info(
        "request_in",
        extra={
            "request_id": request_id,
            "session_id": session_id,
            "msg_len": len(message),
        },
    )

    with timing.request_timing(request_id=request_id, session_id=session_id) as pt:
        loaded = pt.run_parallel(
            "handler.ddb_load",
            {
                "ddb.get_history": lambda: history_store.get_history(
                    session_id, limit=config.HISTORY_LIMIT
                ),
                "ddb.get_session_state": lambda: history_store.get_session_state(session_id),
            },
        )
        history = loaded["ddb.get_history"]
        session_state = loaded["ddb.get_session_state"]

        with timing.stage("router.handle"):
            result = router.handle(message, session_state, history)

        menu_del_dia: list = session_state.get("menu_del_dia", [])
        if result.save_to_menu and result.menu_entry:
            name_en = result.menu_entry.get("name_en", "")
            menu_del_dia = (
                [e for e in menu_del_dia if e.get("name_en") != name_en]
                if name_en else menu_del_dia
            )
            menu_del_dia = menu_del_dia + [result.menu_entry]

        if result.intent == "traduccion":
            if result.save_to_menu:
                new_state: dict = {
                    "menu_del_dia": menu_del_dia,
                    "current_dish": None,
                    "companions": [],
                    "dish_status": None,
                    "collected_ingredients": [],
                    "detected_flags": [],
                    "current_dishes": [],
                }
            else:
                new_state = {
                    "menu_del_dia": menu_del_dia,
                    "current_dish": result.current_dishes[0] if result.current_dishes else None,
                    "companions": list(result.current_dishes[1:]),
                    "dish_status": result.dish_status,
                    "collected_ingredients": list(result.collected_ingredients or []),
                    "detected_flags": list(result.detected_flags or []),
                    "current_dishes": list(result.current_dishes),
                }
        else:
            # Non-traduccion: preserve existing dish state
            new_state = {
                "menu_del_dia": menu_del_dia,
                "current_dish": session_state.get("current_dish"),
                "companions": session_state.get("companions", []),
                "dish_status": session_state.get("dish_status"),
                "collected_ingredients": session_state.get("collected_ingredients", []),
                "detected_flags": session_state.get("detected_flags", []),
                "current_dishes": session_state.get("current_dishes", []),
            }
        with timing.stage("ddb.set_session_state"):
            history_store.set_session_state(session_id, new_state)

        with timing.stage("ddb.append_turns"):
            history_store.append_turns(session_id, [
                {"role": "user", "text": message},
                {"role": "agent", "text": "\n\n".join(result.response)},
            ])

        pt.log_summary(intent=result.intent)

        logger.info(
            "request_out",
            extra={
                "request_id": request_id,
                "session_id": session_id,
                "bubbles": len(result.response),
                "buttons": len(result.buttons),
                "current_dishes_out": result.current_dishes,
                "total_ms": round(pt.total_ms, 2),
            },
        )

        return _response(200, {
            "response": result.response,
            "current_dishes": result.current_dishes,
            "buttons": result.buttons,
            "flags": result.flags,
            "menu_del_dia": menu_del_dia,
            "intent": result.intent,
            "links": result.links,
        })


def _parse_body(event: dict[str, Any]) -> dict[str, Any]:
    raw = event.get("body")
    if raw is None:
        raise _BadRequest("body vacío")
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise _BadRequest("body con tipo inválido")
    if event.get("isBase64Encoded"):
        try:
            raw = base64.b64decode(raw).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as e:
            raise _BadRequest(f"body base64 inválido: {e}") from e
    if len(raw.encode("utf-8")) > _MAX_BODY_BYTES:
        raise _BadRequest("body excede tamaño máximo")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise _BadRequest(f"JSON inválido: {e}") from e
    if not isinstance(data, dict):
        raise _BadRequest("body debe ser un objeto JSON")
    return data


def _require_str(body: dict[str, Any], key: str) -> str:
    val = body.get(key)
    if not isinstance(val, str) or not val.strip():
        raise _BadRequest(f"{key} requerido")
    return val.strip() if key == "session_id" else val


def _optional_list(body: dict[str, Any], key: str) -> list[str]:
    val = body.get(key)
    if val is None:
        return []
    if not isinstance(val, list):
        raise _BadRequest(f"{key} debe ser un array")
    result: list[str] = []
    for item in val:
        if isinstance(item, str) and item.strip():
            result.append(item.strip().lower())
    return result


def _is_preflight(event: dict[str, Any]) -> bool:
    method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod")
        or ""
    )
    return method.upper() == "OPTIONS"



def _response(status: int, body: dict[str, Any] | None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json; charset=utf-8", **_CORS_HEADERS}
    return {
        "statusCode": status,
        "headers": headers,
        "body": "" if body is None else json.dumps(body, ensure_ascii=False),
    }
