# fonda-backend — Arquitectura y Flujo de Trabajo

## ¿Qué es este proyecto?

API serverless que ayuda a fonderos a **traducir platillos mexicanos al inglés** para su menú. Usa IA conversacional (AWS Bedrock) para guiar al usuario a través de las etapas de descripción y traducción.

---

## Stack tecnológico

| Capa | Tecnología |
|------|-----------|
| Runtime | Python 3.10+ |
| Compute | AWS Lambda |
| LLM | AWS Bedrock (Amazon Nova Lite) |
| Base de datos | AWS DynamoDB |
| CI/CD | GitHub Actions |
| Testing | pytest (con mocks de AWS) |

---

## Estructura del proyecto

```
fonda-backend/
├── handler.py          ← Entrada AWS Lambda (HTTP)
├── router.py           ← Orquestador del pipeline
├── classifier.py       ← Clasificación de intención via LLM
├── generation.py       ← Generación de respuesta via LLM
├── retrieval.py        ← Lectura del Knowledge Base
├── bedrock_client.py   ← Cliente AWS Bedrock (reintentos, parsing JSON)
├── history_store.py    ← Persistencia en DynamoDB
├── prompt_loader.py    ← Carga archivos de prompts
├── config.py           ← Variables de entorno y constantes
│
├── kb/                 ← Knowledge Base
│   ├── entities_index.json      ← Mapa alias → entidad canónica
│   ├── platillos/               ← Docs por platillo (mole.txt, arroz.txt…)
│   ├── higiene.txt              ← Guía de buenas prácticas en cocina
│   └── maps.txt                 ← Guía de registro en Tripadvisor
│
├── prompts/
│   ├── classifier_system.txt    ← System prompt del clasificador
│   └── generation_system.txt    ← System prompt del generador
│
├── tests/
│   ├── test_flow.py             ← Tests end-to-end (Bedrock mockeado)
│   └── test_retrieval.py        ← Tests del Knowledge Base
│
└── .github/workflows/deploy.yml ← CI/CD → Lambda
```

---

## Arquitectura general

```
┌─────────────────────────────────────────────────────────────┐
│                        AWS Lambda                           │
│                                                             │
│  HTTP POST          ┌──────────┐    ┌─────────────────────┐ │
│  ──────────────────►│ handler  │    │     DynamoDB        │ │
│                     │    .py   │◄──►│  (historial de      │ │
│  HTTP Response      └────┬─────┘    │   conversación)     │ │
│  ◄──────────────────     │          └─────────────────────┘ │
│                          ▼                                  │
│                   ┌──────────────┐                          │
│                   │  router.py   │                          │
│                   │ (orquestador)│                          │
│                   └──────┬───────┘                          │
│                          │                                  │
│            ┌─────────────┼──────────────┐                   │
│            ▼             ▼              ▼                   │
│    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│    │ classifier   │ │  retrieval   │ │  generation  │       │
│    │    .py       │ │    .py       │ │    .py       │       │
│    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘       │
│           │                │                │               │
└───────────┼────────────────┼────────────────┼───────────────┘
            ▼                ▼                ▼
     ┌──────────────┐  ┌──────────┐   ┌──────────────┐
     │  AWS Bedrock │  │    kb/   │   │  AWS Bedrock │
     │ (Nova Lite)  │  │ (archivos│   │ (Nova Lite)  │
     │ classifier   │  │  .txt)   │   │ generation   │
     └──────────────┘  └──────────┘   └──────────────┘
```

---

## Flujo de trabajo completo

### Paso a paso por petición

```
1. Usuario envía mensaje
   POST / { session_id, message, current_dishes[] }
            │
            ▼
2. handler.py
   ├─ Valida CORS (OPTIONS → 204)
   ├─ Parsea JSON (soporta base64)
   ├─ Carga historial de DynamoDB (últimas 20 turns)
   └─ Llama a router.handle()
            │
            ▼
3. classifier.classify()   [Bedrock, temp=0.0, max=512 tokens]
   ├─ Recibe: message + current_dishes + historial
   ├─ Detecta intención: "traduccion" | "maps" | "higiene" | "fallback"
   ├─ Extrae platillos mencionados (acumulados en current_dishes)
   ├─ Detecta si usuario pidió traducir ya (translate_now)
   └─ Detecta si falta definir variante (pending_variant_for)
            │
            ▼
4. retrieval.get_context()
   ├─ Si intent="traduccion" → lee kb/platillos/{entidad}.txt
   ├─ Si intent="maps"       → lee kb/maps.txt
   ├─ Si intent="higiene"    → lee kb/higiene.txt
   └─ Si intent="fallback"   → sin contexto
            │
            ▼
5. generation.generate()    [Bedrock, temp=0.5, max=1200 tokens]
   ├─ Decide en qué ETAPA está la conversación:
   │   ├─ ETAPA A: Preguntar variante (si pending_variant_for ≠ null)
   │   │           → botones con variantes disponibles
   │   ├─ ETAPA B: Describir en español (si platillo listo, sin traducir aún)
   │   │           → botón "Traducir al inglés"
   │   └─ ETAPA C: Traducir al inglés (si translate_now=true)
   │               → limpia current_dishes (conversación completa)
   └─ Retorna: response[], current_dishes[], buttons[]
            │
            ▼
6. history_store.append_turns()
   └─ Guarda { role: "user", text: ... } y { role: "agent", text: ... } en DynamoDB
            │
            ▼
7. Respuesta HTTP 200
   { response: ["...","..."], current_dishes: [...], buttons: [...] }
```

---

## Etapas de la conversación (Traducción)

El flujo de traducción tiene **4 etapas** bien definidas:

```
Usuario: "quiero poner mole en mi menú"
                    │
                    ▼
           ┌─────────────────┐
           │ ¿Tiene variantes?│
           └────────┬────────┘
                    │ Sí → ETAPA A
                    ▼
     "¿Qué tipo de mole quieres?"
     [Botones: Mole Poblano | Mole Negro | Mole Verde]
                    │
                    ▼ Usuario elige variante
                    │
              ETAPA B ──────────────────────────────┐
     "El Mole Poblano es una salsa tradicional..."   │
     [Botón: Traducir al inglés]                     │
                    │                                │
                    ▼ Usuario pide traducción         │
                    │                                │
              ETAPA C                                │
     "Mole Poblano: Traditional Mexican sauce..."    │
     current_dishes se vacía ─────────────────────── ┘
```

---

## Intenciones soportadas

| Intención | Trigger | KB usado | Comportamiento |
|-----------|---------|----------|---------------|
| `traduccion` | Platillo mencionado | `kb/platillos/*.txt` | Flujo 4 etapas |
| `maps` | Tripadvisor, mapas, reseñas | `kb/maps.txt` | Respuesta directa |
| `higiene` | Limpieza, sanidad, COFEPRIS | `kb/higiene.txt` | Respuesta directa |
| `fallback` | Cualquier otra cosa | Ninguno | Mensaje genérico |

---

## Modelos de datos

### Request / Response HTTP

```python
# Request
{
  "session_id": "usuario-123",        # Identificador de sesión
  "message": "quiero traducir mole",  # Mensaje del usuario
  "current_dishes": ["mole"]          # Estado acumulado (opcional)
}

# Response
{
  "response": ["Globo 1", "Globo 2"], # Burbujas de chat
  "current_dishes": ["mole"],         # Estado persistido para siguiente turno
  "buttons": ["Mole Poblano", "..."]  # Botones de respuesta rápida
}
```

### DynamoDB Schema

```python
{
  "session_id": "usuario-123",
  "turns": [
    { "role": "user",  "text": "...", "ts": 1716220000 },
    { "role": "agent", "text": "...", "ts": 1716220001 }
  ]
}
```

---

## Variables de entorno

| Variable | Requerida | Descripción |
|----------|-----------|-------------|
| `NOVA_LITE_MODEL_ID` | ✅ | ID del modelo Bedrock (`us.amazon.nova-lite-v1:0`) |
| `DDB_TABLE_NAME` | ✅ | Nombre de la tabla DynamoDB |
| `AWS_REGION` | ❌ | Región para Bedrock (default: `us-east-1`) |
| `DDB_REGION` | ❌ | Región para DynamoDB (default: `us-east-1`) |
| `KB_PATH` | ❌ | Ruta al Knowledge Base (default: `./kb`) |
| `CORS_ALLOW_ORIGIN` | ❌ | Origen CORS permitido (default: `*`) |

---

## CI/CD

```
git push → main
    │
    ▼
GitHub Actions (.github/workflows/deploy.yml)
    ├─ zip *.py + kb/ + prompts/
    └─ aws lambda update-function-code --function-name prod-menu-del-dia
```

El deploy es automático en cada push a `main`. También se puede disparar manualmente vía `workflow_dispatch`.

---

## Testing

```bash
# Todos los tests
pytest

# Tests de flujo end-to-end (Bedrock mockeado)
pytest tests/test_flow.py

# Tests del Knowledge Base
pytest tests/test_retrieval.py
```

Los tests no requieren credenciales AWS — `conftest.py` mockea `boto3` y `botocore` antes de que los módulos los importen.

---

> El proyecto es deliberadamente simple: **un único endpoint Lambda** que orquesta dos llamadas a Bedrock (clasificar + generar) con contexto del Knowledge Base, persistiendo el historial en DynamoDB para mantener la conversación.
