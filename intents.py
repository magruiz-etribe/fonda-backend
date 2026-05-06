from __future__ import annotations

from typing import Final

INTENTS: Final[dict[str, str]] = {
    "traducir": (
        "El usuario quiere traducir uno o varios platillos al inglés, "
        "describir su contenido en inglés y declarar alérgenos. "
        "Aplica también si el usuario manda foto de un platillo o de un menú impreso."
    ),
    "maps": (
        "El usuario pide ayuda para registrar o configurar su negocio en Google Maps "
        "(crear ficha, verificación, fotos, horarios, reseñas)."
    ),
    "higiene": (
        "El usuario pregunta por buenas prácticas de higiene en su cocina o fonda "
        "(manejo de alimentos, limpieza, manejo de basura, lavado de manos)."
    ),
    "fallback": (
        "Cualquier otra cosa fuera de los temas anteriores "
        "(clima, política, charla casual, otros temas)."
    ),
}
