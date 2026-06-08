from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Final

import config
import history_store
import router
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
        current_dishes = _optional_list(body, "current_dishes")
    except _BadRequest as e:
        logger.warning("bad_request", extra={"request_id": request_id, "error": str(e)})
        return _response(400, {"error": str(e)})

    logger.info(
        "request_in",
        extra={
            "request_id": request_id,
            "session_id": session_id,
            "msg_len": len(message),
            "current_dishes_in": current_dishes,
        },
    )

    history = history_store.get_history(session_id, limit=config.HISTORY_LIMIT)

    session_state = history_store.get_session_state(session_id)
    if "current_dishes" in session_state:
        current_dishes = session_state["current_dishes"]

    _CONF_KEYS = ("completeness_confirmed", "allergens_confirmed", "gluten_confirmed", "spicy_confirmed")
    confirmation_state = {k: session_state.get(k) for k in _CONF_KEYS}
    dish_context: dict = session_state.get("dish_context") or {}

    result = router.handle(message, current_dishes, history, confirmation_state, dish_context or None)

    # Merge new confirmations from this turn into the running state
    prev_primary = (session_state.get("current_dishes") or [""])[0]
    curr_primary = (result.current_dishes or [""])[0]
    primary_changed = bool(curr_primary) and curr_primary != prev_primary

    if result.current_dishes and not primary_changed:
        merged_conf: dict = {k: v for k, v in confirmation_state.items() if v is not None}
    else:
        merged_conf = {}  # reset when dish flow ends or primary dish changes

    for key, val in [
        ("completeness_confirmed", result.completeness_confirmed),
        ("allergens_confirmed", result.allergens_confirmed),
        ("gluten_confirmed", result.gluten_confirmed),
        ("spicy_confirmed", result.spicy_confirmed),
    ]:
        if val is not None:
            merged_conf[key] = val

    menu_del_dia: list = session_state.get("menu_del_dia", [])
    if result.menu_entry:
        result.menu_entry["flags"] = _apply_flag_confirmations(result.menu_entry["flags"], merged_conf)
        name_en = result.menu_entry.get("name_en", "")
        menu_del_dia = [e for e in menu_del_dia if e.get("name_en") != name_en] if name_en else menu_del_dia
        menu_del_dia = menu_del_dia + [result.menu_entry]

    # Update dish_context
    if primary_changed or result.menu_entry:
        # New dish or translation completed — reset dish_context
        dish_context = {}

    if result.current_dishes:
        dish_context["main_dish"] = curr_primary
        # Accumulate resolved_variants (LLM values take priority)
        persisted_rv: dict = dish_context.get("resolved_variants") or {}
        dish_context["resolved_variants"] = {**persisted_rv, **result.resolved_variants}
        # Accumulate extra_ingredients (deduplicated)
        existing_extras: list = dish_context.get("extra_ingredients") or []
        for ing in result.extra_user_ingredients:
            if ing not in existing_extras:
                existing_extras.append(ing)
        dish_context["extra_ingredients"] = existing_extras
        # Capture last Spanish description (when "✅ Adaptar al inglés" button is offered)
        if "✅ Adaptar al inglés" in result.buttons:
            last_response = "\n\n".join(result.response)
            dish_context["last_description_es"] = last_response

    new_state: dict = {"current_dishes": result.current_dishes, "menu_del_dia": menu_del_dia, "dish_context": dish_context}
    new_state.update(merged_conf)
    history_store.set_session_state(session_id, new_state)

    history_store.append_turns(session_id, [
        {"role": "user", "text": message},
        {"role": "agent", "text": "\n\n".join(result.response)},
    ])

    logger.info(
        "request_out",
        extra={
            "request_id": request_id,
            "session_id": session_id,
            "bubbles": len(result.response),
            "buttons": len(result.buttons),
            "current_dishes_out": result.current_dishes,
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


def _apply_flag_confirmations(flags: dict, conf: dict) -> dict:
    """Override clean flags based on what the fondero confirmed."""
    flags = dict(flags)
    if conf.get("allergens_confirmed") is False:
        flags["allergens"] = False
    if conf.get("gluten_confirmed") is False:
        flags["gluten_free"] = True
    if conf.get("spicy_confirmed") is False:
        flags["spicy_level"] = "none"
    return flags


def _response(status: int, body: dict[str, Any] | None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json; charset=utf-8", **_CORS_HEADERS}
    return {
        "statusCode": status,
        "headers": headers,
        "body": "" if body is None else json.dumps(body, ensure_ascii=False),
    }
