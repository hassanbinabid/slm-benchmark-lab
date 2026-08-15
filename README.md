# Local AI Assistant

> A production-pattern offline AI assistant built on small language models running entirely on local hardware. No cloud dependency, no API costs, no data leaving the machine.

---

## Why This Project

Most LLM tutorials show you how to call an API. This project shows you how to **engineer around one** — measuring performance rigorously, constraining outputs reliably, and comparing models with evidence rather than opinion.

Built to demonstrate the engineering skills that matter in production AI systems:
- Rigorous performance measurement
- Constrained generation with schema validation
- Retry logic and graceful failure handling
- Multi-model benchmarking with quantization analysis
- Production REST API with FastAPI

---

## Project Structure

```
local-ai-assistant/
├── benchmark/
│   ├── benchmark.py          # Phase 1 — inference benchmarking
│   └── quant_check.py        # Quantization format investigation
├── structured/
│   └── structured_gen.py     # Phase 2 — JSON schema + Pydantic + retry
├── comparison/
│   └── model_compare.py      # Phase 3 — multi-model evaluation
├── api/
│   └── main.py               # FastAPI wrapper with 5 endpoints
├── results/
│   ├── benchmark_results.csv
│   ├── structured_results.csv
│   └── model_comparison_scored.csv
└── requirements.txt
```

---

## Quick Start

### 1 — Install Ollama
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download from https://ollama.com/download
```

### 2 — Pull models
```bash
ollama pull llama3.2:3b
ollama pull phi4-mini
ollama pull mistral:7b
```

### 3 — Install Python dependencies
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 4 — Run each phase
```bash
# Phase 1 — Benchmarking
python benchmark/benchmark.py

# Phase 2 — Structured generation
python structured/structured_gen.py

# Phase 3 — Model comparison (takes 30–45 min)
python comparison/model_compare.py

# FastAPI server
uvicorn api.main:app --reload --port 8000
# Then open http://localhost:8000/docs
```

---

## Phase 1 — Inference Benchmarking

Measures three key metrics on `llama3.2:3b` across 10 prompts:

| Metric | Result |
|---|---|
| Avg Time to First Token | 3,813 ms |
| Avg Total Latency | 21.4 s |
| Sustained Throughput | 16–17 tok/s |
| Hardware | CPU-only, Windows 11 |

**Key finding:** Cold start TTFT is 2.4x higher than warm TTFT (8,314ms vs 3,400ms). In production, always warmup the model at server startup so users never feel first-load latency.

---

## Phase 2 — Structured Output Engineering

Implements the **constraint → validate → retry** pattern used in production AI pipelines.

```python
class ConceptExplanation(BaseModel):
    topic: str
    difficulty: str      # must be: easy / intermediate / advanced
    explanation: str     # max 80 words
    has_code_example: bool
    one_line_summary: str  # max 20 words
```

**Temperature experiment results:**

| Temperature | Success rate | Finding |
|---|---|---|
| 0.0 (deterministic) | 1/5 | Markdown wrapping caused JSON parse failures |
| 0.7 (stochastic) | 5/5 | Retry mechanism recovered all failures |

**Key finding:** For small models, temperature 0.0 can *reduce* instruction-following reliability. A small amount of randomness helped the model comply with formatting instructions. Always test structured generation at multiple temperatures.

---

## Phase 3 — Model Comparison Study

Three models evaluated on 40 standardised prompts across 5 task categories.

### Speed

| Model | Avg Tok/s | Avg Latency | Avg Word Count |
|---|---|---|---|
| llama3.2:3b | **14.55** | 9.8s | 116 |
| mistral:7b | 7.42 | 23.6s | 119 |
| phi4-mini | 5.90 | 85.2s | **772** |

### Quality (scored 1–3)

| Model | Overall | Factual | Reasoning | Code | Instruction |
|---|---|---|---|---|---|
| mistral:7b | **2.92** | 3.00 | **2.88** | **3.00** | 2.75 |
| llama3.2:3b | 2.85 | 3.00 | 2.50 | 2.75 | **3.00** |
| phi4-mini | 1.92 | 3.00 | 1.25 | 1.00 | 1.62 |

**Key findings:**
- `mistral:7b` is the best quality model at 2.92/3.0 — worth the slower speed for quality-critical tasks
- `llama3.2:3b` is the best speed-quality tradeoff — only 0.07 below mistral at nearly 2x the speed
- `phi4-mini` generated 6.6x more words than the other models yet scored lowest — verbosity is not a proxy for quality
- All three models scored equally on factual recall — model size only matters for reasoning and generation tasks

---

## Quantization Analysis

A post-hoc investigation revealed that all models pulled via Ollama are already distributed as **GGUF Q4_K_M** format by default.
This means the benchmark results already reflect quantized performance. Full F32 precision would require ~6x more memory (12 GB for llama, 28 GB for mistral) with negligible quality benefit on CPU hardware.

**Key finding:** The most important quantization decision in production is not which level to choose — Ollama makes the right default. The critical decision is which model size fits your hardware constraints.

---

## FastAPI Endpoints

Start the server:
```bash
uvicorn api.main:app --reload --port 8000
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Ollama connectivity + available models |
| `/models` | GET | List all local models with sizes |
| `/generate` | POST | Text generation with timing metrics |
| `/benchmark` | POST | Full TTFT + latency + tok/s measurement |
| `/structured` | POST | JSON-enforced generation with retry logic |

Interactive docs: `http://localhost:8000/docs`

---

## Key Engineering Learnings

| Finding | Production Implication |
|---|---|
| Cold start TTFT is 2.4x higher than warm | Always warmup model at server startup |
| Latency scales linearly with output length | Always set max_tokens limits for SLA compliance |
| Temperature 0.0 reduced JSON compliance | Test structured generation at multiple temperatures |
| Retry with error context achieved 100% recovery | Include exact validation error in retry prompts |
| Pydantic caught all malformed responses | Never pass raw LLM output into application logic |
| mistral:7b best quality at 2.92/3.0 | Larger models justify cost for reasoning and code tasks |
| phi4-mini: 6.6x more words, lower quality | Verbosity is not a proxy for quality |
| Ollama default pulls are already Q4_K_M | No manual quantization needed for CPU deployment |

---

## Hardware

All benchmarks run on:
- **OS:** Windows 11
- **Inference:** CPU-only (no GPU)
- **Runtime:** Ollama 0.20.5

---

## License

MIT — use freely for your portfolio.
