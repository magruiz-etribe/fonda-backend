from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Final, Literal

CUSTOM_ENTITY: Final[str] = "__custom__"

_SCHEMA_VERSION: Final[int] = 1

ImageType = Literal["platillo", "menu"]
Mode = Literal["platillo", "menu"]


@dataclass
class CurrentDish:
    entity: str
    variant: str | None = None
    user_ingredients: list[str] = field(default_factory=list)


@dataclass
class CompletedDish:
    dish: str
    ingredients: list[str] = field(default_factory=list)
    translation_en: str = ""
    description_en: str = ""
    allergens: list[str] = field(default_factory=list)
    approved: bool | None = None


@dataclass
class ImageAnalysis:
    type: ImageType
    raw: str
    detected_dishes: list[str] = field(default_factory=list)


@dataclass
class PausedTask:
    intent: str | None = None
    mode: Mode | None = None
    current_dish: CurrentDish | None = None
    dish_queue: list[str] = field(default_factory=list)


@dataclass
class AgentState:
    v: int = _SCHEMA_VERSION
    intent: str | None = None
    mode: Mode | None = None
    current_dish: CurrentDish | None = None
    dish_queue: list[str] = field(default_factory=list)
    completed: list[CompletedDish] = field(default_factory=list)
    paused_task: PausedTask | None = None
    image_analysis: ImageAnalysis | None = None


def init_state() -> AgentState:
    return AgentState()


def to_dict(state: AgentState) -> dict[str, Any]:
    return asdict(state)


def from_dict(data: dict[str, Any] | None) -> AgentState:
    if not data:
        return init_state()

    return AgentState(
        v=int(data.get("v", _SCHEMA_VERSION)),
        intent=data.get("intent"),
        mode=_coerce_mode(data.get("mode")),
        current_dish=_current_dish_from_dict(data.get("current_dish")),
        dish_queue=_str_list(data.get("dish_queue")),
        completed=[_completed_from_dict(c) for c in (data.get("completed") or [])],
        paused_task=_paused_from_dict(data.get("paused_task")),
        image_analysis=_image_from_dict(data.get("image_analysis")),
    )


def validate_version(state: AgentState) -> AgentState:
    # If a future client sends a newer schema we drop unknown structure by
    # re-emitting from_dict(to_dict(...)) of a default; for now we only know v=1.
    if state.v != _SCHEMA_VERSION:
        return init_state()
    return state


def pause_current(state: AgentState) -> None:
    if state.intent is None and state.current_dish is None and not state.dish_queue:
        return
    candidate = PausedTask(
        intent=state.intent,
        mode=state.mode,
        current_dish=state.current_dish,
        dish_queue=list(state.dish_queue),
    )
    prev = state.paused_task
    # No pisar una pausa valiosa de traducción (platillo + ingredientes) con un snap
    # vacío cuando el clasificador hace zig-zag ej. traducir → fallback → traducir.
    trash_snap = (
        candidate.current_dish is None
        and not candidate.dish_queue
        and candidate.intent != "traducir"
    )
    keep_prior = (
        prev is not None
        and prev.intent == "traducir"
        and prev.current_dish is not None
        and trash_snap
    )
    if not keep_prior:
        state.paused_task = candidate
    state.mode = None
    state.current_dish = None
    state.dish_queue = []


def resume_paused(state: AgentState) -> str | None:
    pt = state.paused_task
    if pt is None:
        return None
    state.intent = pt.intent
    state.mode = pt.mode
    state.current_dish = pt.current_dish
    state.dish_queue = list(pt.dish_queue)
    state.paused_task = None

    if pt.current_dish is not None:
        return (
            f"Por cierto, dejamos pendiente la traducción de "
            f"'{pt.current_dish.entity}'. ¿La retomamos?"
        )
    if pt.dish_queue:
        return "Aún teníamos platillos pendientes del menú. ¿Continuamos con esos?"
    if pt.intent:
        return f"Habíamos quedado en algo de '{pt.intent}'. ¿Continuamos?"
    return None


def _coerce_mode(value: Any) -> Mode | None:
    if value in ("platillo", "menu"):
        return value
    return None


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(x) for x in value if isinstance(x, (str, int, float))]


def _current_dish_from_dict(d: Any) -> CurrentDish | None:
    if not isinstance(d, dict):
        return None
    entity = d.get("entity")
    if not isinstance(entity, str) or not entity:
        return None
    return CurrentDish(
        entity=entity,
        variant=d.get("variant") if isinstance(d.get("variant"), str) else None,
        user_ingredients=_str_list(d.get("user_ingredients")),
    )


def _completed_from_dict(d: Any) -> CompletedDish:
    if not isinstance(d, dict):
        return CompletedDish(dish="")
    approved = d.get("approved")
    if approved is not None and not isinstance(approved, bool):
        approved = None
    return CompletedDish(
        dish=str(d.get("dish", "")),
        ingredients=_str_list(d.get("ingredients")),
        translation_en=str(d.get("translation_en", "")),
        description_en=str(d.get("description_en", "")),
        allergens=_str_list(d.get("allergens")),
        approved=approved,
    )


def _paused_from_dict(d: Any) -> PausedTask | None:
    if not isinstance(d, dict):
        return None
    return PausedTask(
        intent=d.get("intent") if isinstance(d.get("intent"), str) else None,
        mode=_coerce_mode(d.get("mode")),
        current_dish=_current_dish_from_dict(d.get("current_dish")),
        dish_queue=_str_list(d.get("dish_queue")),
    )


def _image_from_dict(d: Any) -> ImageAnalysis | None:
    if not isinstance(d, dict):
        return None
    img_type = d.get("type")
    if img_type not in ("platillo", "menu"):
        return None
    return ImageAnalysis(
        type=img_type,
        raw=str(d.get("raw", "")),
        detected_dishes=_str_list(d.get("detected_dishes")),
    )
