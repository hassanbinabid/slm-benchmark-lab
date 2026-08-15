"""
FastAPI wrapper around Ollama local inference.
Exposes clean REST endpoints for generation, benchmarking, and structured output.
Run: uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import json
import time
import statistics
from typing import Optional

import httpx
import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError, field_validator

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Local AI Assistant API",
    description="Production-pattern local inference API using Ollama. No cloud, no API costs, no data leaving the machine.",
    version="1.0.0",
)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL   = "llama3.2:3b"


# ── Request / Response models ─────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = DEFAULT_MODEL
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500

class GenerateResponse(BaseModel):
    model: str
    prompt: str
    response: str
    tokens_per_second: float
    total_latency_s: float

class BenchmarkRequest(BaseModel):
    prompt: str
    model: Optional[str] = DEFAULT_MODEL

class BenchmarkResponse(BaseModel):
    model: str
    prompt: str
    ttft_ms: float
    total_latency_s: float
    output_tokens: int
    tokens_per_second: float

class StructuredRequest(BaseModel):
    topic: str
    model: Optional[str] = DEFAULT_MODEL
    temperature: Optional[float] = 0.7

class ConceptExplanation(BaseModel):
    topic: str
    difficulty: str
    explanation: str
    has_code_example: bool
    one_line_summary: str

    @field_validator("difficulty")
    @classmethod
    def difficulty_must_be_valid(cls, v):
        allowed = {"easy", "intermediate", "advanced"}
        if v.lower() not in allowed:
            raise ValueError(f"difficulty must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("one_line_summary")
    @classmethod
    def summary_must_be_short(cls, v):
        if len(v.split()) > 20:
            raise ValueError(f"one_line_summary too long (max 20 words)")
        return v

# ── Helper functions ──────────────────────────────────────────────────────────
def check_ollama_running() -> bool:
    """Check if Ollama is reachable."""
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def run_inference(prompt: str, model: str, temperature: float, max_tokens: int) -> dict:
    """
    Run inference and return response + timing metrics.
    Uses streaming to capture TTFT separately from total latency.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    first_token_time = None
    output_tokens    = 0
    full_response    = ""
    start_time       = time.perf_counter()

    with httpx.stream("POST", url, json=payload, timeout=120) as response:
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)

            if first_token_time is None and chunk.get("response"):
                first_token_time = time.perf_counter()

            full_response += chunk.get("response", "")

            if chunk.get("done"):
                output_tokens = chunk.get("eval_count", 0)
                break

    end_time        = time.perf_counter()
    ttft_ms         = (first_token_time - start_time) * 1000 if first_token_time else 0
    total_latency_s = end_time - start_time
    tokens_per_sec  = output_tokens / total_latency_s if total_latency_s > 0 else 0

    return {
        "response":          full_response.strip(),
        "ttft_ms":           round(ttft_ms, 2),
        "total_latency_s":   round(total_latency_s, 3),
        "output_tokens":     output_tokens,
        "tokens_per_second": round(tokens_per_sec, 2),
    }


def build_structured_prompt(topic: str, previous_error: str = None) -> str:
    """Build a constrained prompt for structured JSON output."""
    prompt = f"""You are a technical educator. Explain the concept: "{topic}"

Respond with ONLY a raw JSON object. No markdown, no code blocks, no extra text.

The JSON must have exactly these fields:
- "topic": string
- "difficulty": string (must be exactly: easy, intermediate, or advanced)
- "explanation": string (plain English, max 80 words)
- "has_code_example": boolean
- "one_line_summary": string (max 20 words)

Example:
{{
    "topic": "recursion",
    "difficulty": "intermediate",
    "explanation": "A function that calls itself to solve smaller versions of a problem.",
    "has_code_example": false,
    "one_line_summary": "A function that calls itself to solve smaller subproblems."
}}"""

    if previous_error:
        prompt += f"""

Your previous response failed validation:
{previous_error}

Fix that specific issue and return the full JSON again."""

    return prompt

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Check if Ollama is running and list available models.
    Always call this first to verify the system is ready.
    """
    if not check_ollama_running():
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start it with: ollama serve"
        )

    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        models = []

    return {
        "status":         "healthy",
        "ollama_running": True,
        "models_available": models,
        "default_model":  DEFAULT_MODEL,
    }


@app.get("/models")
def list_models():
    """List all locally available Ollama models with size information."""
    if not check_ollama_running():
        raise HTTPException(status_code=503, detail="Ollama is not running.")

    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = r.json().get("models", [])
        return {
            "models": [
                {
                    "name":     m["name"],
                    "size_gb":  round(m.get("size", 0) / 1e9, 2),
                    "modified": m.get("modified_at", "unknown"),
                }
                for m in models
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """
    Basic text generation endpoint.
    Send a prompt, get a response with timing metrics.
    """
    if not check_ollama_running():
        raise HTTPException(status_code=503, detail="Ollama is not running.")

    try:
        result = run_inference(
            prompt=request.prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return GenerateResponse(
            model=request.model,
            prompt=request.prompt,
            response=result["response"],
            tokens_per_second=result["tokens_per_second"],
            total_latency_s=result["total_latency_s"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/benchmark", response_model=BenchmarkResponse)
def benchmark(request: BenchmarkRequest):
    """
    Benchmark a single prompt and return full timing metrics.
    Runs at temperature=0.0 for deterministic, reproducible results.
    """
    if not check_ollama_running():
        raise HTTPException(status_code=503, detail="Ollama is not running.")

    try:
        result = run_inference(
            prompt=request.prompt,
            model=request.model,
            temperature=0.0,
            max_tokens=500,
        )
        return BenchmarkResponse(
            model=request.model,
            prompt=request.prompt,
            ttft_ms=result["ttft_ms"],
            total_latency_s=result["total_latency_s"],
            output_tokens=result["output_tokens"],
            tokens_per_second=result["tokens_per_second"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/structured")
def structured_generation(request: StructuredRequest):
    """
    JSON-enforced generation with Pydantic validation and retry logic.
    Implements the constraint -> validate -> retry pattern.
    Returns a validated ConceptExplanation object or a failure record.
    """
    if not check_ollama_running():
        raise HTTPException(status_code=503, detail="Ollama is not running.")

    attempt        = 1
    previous_error = None

    while attempt <= 2:
        prompt = build_structured_prompt(request.topic, previous_error)

        try:
            response = ollama.chat(
                model=request.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": request.temperature, "seed": 42},
            )
            raw = response["message"]["content"].strip()

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

        # Try JSON parse
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            previous_error = f"Response was not valid JSON: {e}"
            attempt += 1
            continue

        # Try Pydantic validation
        try:
            validated = ConceptExplanation(**parsed)
            return {
                "status":           "success",
                "attempt":          attempt,
                "topic":            validated.topic,
                "difficulty":       validated.difficulty,
                "explanation":      validated.explanation,
                "has_code_example": validated.has_code_example,
                "one_line_summary": validated.one_line_summary,
            }
        except ValidationError as e:
            previous_error = str(e.errors()[0]["msg"])
            attempt += 1

    # Both attempts failed
    return {
        "status":  "failed",
        "attempt": 2,
        "topic":   request.topic,
        "error":   previous_error,
    }