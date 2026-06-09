# Flujo de Adaptación / Traducción de Menú

Describe paso a paso cómo Huevito convierte el nombre de un platillo en una descripción bilingüe lista para menú.

---

## Arquitectura general

```
Usuario
  │
  ▼
handler.py          ← punto de entrada Lambda; lee/escribe session state en DynamoDB
  │
  ▼
router.handle()     ← orquesta todo el flujo
  ├─ classifier.py  ← identifica intención, platillos, ingredientes extra, slots pendientes
  ├─ flag_llm.py    ← detecta alérgenos / gluten / picor vía LLM
  ├─ flags.py       ← fallback determinista de detección (KB)
  ├─ retrieval.py   ← carga datos del KB (YAML de platillos, índice de entidades)
  └─ generation.py  ← genera respuesta final vía LLM (Amazon Nova Pro)
```

**Estado persistente en DynamoDB** (por sesión):

| Campo | Tipo | Descripción |
|---|---|---|
| `current_dishes` | `list[str]` | Platillos en contexto |
| `completeness_confirmed` | `bool \| null` | ¿El fondero confirmó que el platillo está completo? |
| `allergens_confirmed` | `bool \| null` | ¿Confirmó la presencia de alérgenos? |
| `gluten_confirmed` | `bool \| null` | ¿Confirmó ingredientes con gluten? |
| `spicy_confirmed` | `bool \| null` | ¿Confirmó ingredientes picantes? |
| `dish_context` | `dict` | Variantes resueltas, ingredientes extra, última descripción ES |

---

## Etapas del flujo

```
 Mensaje del usuario
       │
       ▼
 ┌─────────────┐
 │  CLASIFICAR │  classifier.py → intent, current_dishes, pending_slots, extra_user_ingredients
 └──────┬──────┘
        │ intent == "traduccion"
        ▼
 ┌──────────────────┐
 │  DETECTAR FLAGS  │  flag_llm + flags.py (KB) → allergen/gluten/spicy triggers
 └──────┬───────────┘
        │
        ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  SHORT CIRCUIT (_try_short_circuit)  — 100% determinista     │
 │                                                              │
 │  pending_slots?  ──sí──▶  SLOT: pedir variante/relleno/etc  │
 │       │                                                      │
 │       no                                                     │
 │       ▼                                                      │
 │  completeness_confirmed == null?                             │
 │    ├─ usuario confirma "listo" ──▶ A1 confirma → siguiente   │
 │    ├─ usuario agrega ingredientes ──▶ A1 confirma → siguiente│
 │    └─ ninguno de los dos ──▶  A1 HAZ PREGUNTA (1 vez)        │
 │       │                                                      │
 │  allergen_triggers && allergens_confirmed == null?           │
 │    ├─ respuesta recibida ──▶ A2 confirma → siguiente         │
 │    └─ no hay respuesta aún ──▶  A2 HAZ PREGUNTA              │
 │       │                                                      │
 │  gluten_triggers && gluten_confirmed == null?  →  igual A3   │
 │  spicy_triggers  && spicy_confirmed  == null?  →  igual A4   │
 │       │                                                      │
 │  Todo resuelto ──▶ None (LLM genera ETAPA B o C)             │
 └──────────────────────────────────────────────────────────────┘
        │
        │  short == None → LLM path
        ▼
 ┌─────────────────┐
 │  generation.py  │  LLM genera ETAPA B (descripción ES) o ETAPA C (traducción EN)
 └──────┬──────────┘
        │
        ▼
 _force_after_llm   ← fuerza campos de confirmación si el LLM los omite
        │
        ▼
 handler.py guarda nuevo estado en DynamoDB → responde al cliente
```

---

## Detalle de cada etapa

### SLOT — Resolver variante ambigua

Se activa cuando el clasificador no puede determinar qué variante del platillo prepara el fondero (e.g. "mole" sin especificar negro/rojo/verde).

- El `classifier.py` emite un `pending_slots` con `entity`, `slot_name` y `options`.
- Se resuelven uno a uno; el siguiente turno muestra el siguiente slot pendiente.
- Mientras hay slots pendientes, **no se avanza a A1**.

---

### A1 — Confirmar completitud del platillo

Pregunta: *"¿Tu platillo lleva proteína, guarnición, salsa especial o algún complemento?"*

- Se hace **una sola vez** (cuando `completeness_confirmed == null`).
- **RAMA CONFIRMA**: el fondero dice "listo" / "eso es todo" → `completeness_confirmed = true` → avanza.
- **RAMA AGREGA**: el fondero agrega ingredientes → `completeness_confirmed = true` → avanza (no repite la pregunta).
- **Acompañamiento KB**: si el fondero menciona un platillo que existe en el KB como acompañamiento (e.g. "lo acompaño de frijoles refritos"), se detecta como `_companion_added`, se fuerza `completeness_confirmed = true` y se avanza.

---

### A2 — Confirmar alérgenos

Pregunta: *"He detectado que tu platillo puede contener: **huevo, queso**. ¿Confirmas?"*

- Solo se activa si `allergen_triggers` es una lista no vacía (detectada por LLM + KB determinístico).
- Respuestas esperadas: `✅ Sí, contiene alguno` / `❌ No, ninguno de esos`.
- Resultado guardado en `allergens_confirmed`.

---

### A3 — Confirmar gluten

Igual que A2 pero para ingredientes con gluten (`gluten_triggers`).  
La palabra "gluten" **no se menciona** al fondero, solo los ingredientes concretos.

---

### A4 — Confirmar picor

Igual que A2 pero para ingredientes picantes (`spicy_triggers`).

---

### ETAPA B — Descripción en español

El LLM genera una card de menú en español con estructura obligatoria:

```
**Nombre del Platillo** [emoji]
Descripción factual en 2-3 oraciones: ingredientes visibles, técnica de cocción, acompañamientos.
```

Seguida del globo: *"¿Te parece bien? Si quieres cambiar algo dime, o presiona el botón 👇"*  
Botón: `✅ Adaptar al inglés`

Reglas clave:
- Un solo card aunque haya múltiples platillos en `current_dishes` (los acompañamientos se integran al cuerpo).
- Sin adjetivos subjetivos ni lenguaje de marketing.
- Solo ingredientes mencionados por el fondero o presentes en el KB.

---

### ETAPA C — Traducción al inglés

Se activa cuando `translate_now = true` (fondero presionó "✅ Adaptar al inglés").

Fuentes de traducción en orden de prioridad:
1. Última descripción ES generada en ETAPA B (guardada en `dish_context.last_description_es`).
2. Descripción del KB para la variante resuelta.
3. Información del historial.

Estructura de salida:

```
**[Dish Name]** [emoji]
English description matching the Spanish card.

---
[Closing message in Spanish]
```

Al terminar, `current_dishes` se resetea a `[]` y se genera un `menu_entry` guardado en `menu_del_dia`.

---

## Detección de flags (alérgenos / gluten / picor)

Se ejecuta **una vez por turno** antes del short circuit, con dos capas:

1. **LLM** (`flag_llm.py`): analiza el nombre del platillo + historial → devuelve triggers y niveles.
2. **Determinista** (`flags.py` + KB): recorre los ingredientes del YAML del platillo (base + variante resuelta + extra del fondero) y los cruza contra `allergens.yaml`, `spicy_markers.yaml`, `vegetarian_markers.yaml`.

El resultado final es la **unión** de ambas fuentes (más conservador: si cualquiera detecta un trigger, se incluye).

---

## Knowledge Base (KB)

Ubicación: `kb/platillos/*.yaml`

Cada archivo YAML tiene:

```yaml
canonical_name: nombre_canónico
common_names:           # aliases reconocidos
category: antojito
base_ingredients: [...]
variants:
  nombre_variante:
    name_es: Nombre en español
    name_en: Name in English
    extra_ingredients: [...]
    technique: "..."
    description_es: >
      ...
    description_en: >
      ...
```

El índice `kb/entities_index.json` mapea aliases en minúsculas → nombre canónico del archivo YAML.  
El clasificador usa los **valores** del índice (nombres canónicos) como lista de entidades válidas que muestra al LLM.

---

## Ejemplo de flujo completo

| Turno | Usuario | Agente | Estado guardado |
|---|---|---|---|
| 1 | "huevo con jamón" | A1: "¿Tu platillo lleva algo más?" | `completeness_confirmed: null` |
| 2 | "lo acompaño de frijoles refritos y totopos" | A2: "He detectado: **huevo, jamón**. ¿Confirmas?" | `completeness_confirmed: true` |
| 3 | "✅ Sí, contiene alguno" | ETAPA B: card en español + botón "✅ Adaptar al inglés" | `allergens_confirmed: true` |
| 4 | "✅ Adaptar al inglés" | ETAPA C: traducción en inglés | `menu_del_dia` actualizado |

---

## Archivos clave

| Archivo | Rol |
|---|---|
| `handler.py` | Punto de entrada Lambda; gestiona sesión DynamoDB |
| `router.py` | Orquestador; short circuit determinista + LLM path |
| `classifier.py` | Clasifica intención, extrae platillos y slots |
| `generation.py` | Genera ETAPA B y C vía LLM (Amazon Nova Pro) |
| `flag_llm.py` | Detección de flags vía LLM |
| `flags.py` | Detección de flags determinista (KB) |
| `retrieval.py` | Carga datos YAML del KB |
| `kb/entities_index.json` | Índice alias → entidad canónica |
| `kb/platillos/*.yaml` | Datos de cada platillo |
| `kb/allergens.yaml` | Grupos de alérgenos y triggers |
| `kb/spicy_markers.yaml` | Niveles de picor y triggers |
| `prompts/generation_system.txt` | System prompt del LLM generador |
