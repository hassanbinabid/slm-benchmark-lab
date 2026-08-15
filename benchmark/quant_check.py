"""
Quantization Analysis
======================
Investigates the quantization format of locally pulled Ollama models.
Documents the finding that Ollama default pulls are already Q4 quantized.
Run: python benchmark/quant_check.py
"""

import httpx
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
OLLAMA_BASE_URL = "http://localhost:11434"


def get_model_info(model: str) -> dict:
    """Fetch model metadata from Ollama API."""
    try:
        r = httpx.post(
            f"{OLLAMA_BASE_URL}/api/show",
            json={"name": model},
            timeout=10
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def get_all_models() -> list:
    """List all locally available models."""
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        return r.json().get("models", [])
    except Exception as e:
        return []


def main():
    console.print(Panel(
        "Investigating quantization format of local Ollama models.",
        title="Quantization Analysis",
        border_style="blue"
    ))

    models = get_all_models()

    # ── Model inventory table ─────────────────────────────────────────────
    table = Table(title="Local Model Inventory", show_lines=True)
    table.add_column("Model", style="cyan", width=38)
    table.add_column("Size (GB)", justify="right", style="yellow")
    table.add_column("ID", style="dim", width=16)

    for m in models:
        size_gb = round(m.get("size", 0) / 1e9, 2)
        table.add_row(
            m["name"],
            str(size_gb),
            m["digest"][:12],
        )

    console.print(table)

    # ── Check quantization details ────────────────────────────────────────
    models_to_check = [
        "llama3.2:3b",
        "llama3.2:3b-instruct-q4_K_M",
        "mistral:7b",
        "mistral:7b-instruct-v0.3-q4_K_M",
    ]

    console.print("\n[bold yellow]── Quantization Details ──[/bold yellow]\n")

    quant_table = Table(title="Model Quantization Info", show_lines=True)
    quant_table.add_column("Model", style="cyan", width=38)
    quant_table.add_column("Quantization", style="green", width=16)
    quant_table.add_column("Format", style="yellow", width=10)
    quant_table.add_column("Parameters", justify="right", width=12)

    for model_name in models_to_check:
        info = get_model_info(model_name)

        details      = info.get("details", {})
        quant_level  = details.get("quantization_level", "unknown")
        format_type  = details.get("format", "unknown")
        param_size   = details.get("parameter_size", "unknown")

        quant_table.add_row(
            model_name,
            quant_level,
            format_type,
            param_size,
        )

    console.print(quant_table)

    # ── Engineering finding ───────────────────────────────────────────────
    console.print(Panel(
        "[bold]Key Finding:[/bold]\n\n"
        "Ollama's default model pulls are already GGUF quantized.\n"
        "The llama3.2:3b and mistral:7b models served by Ollama are\n"
        "not full F32 precision — they are pre-quantized for CPU inference.\n\n"
        "[bold]Implication:[/bold]\n\n"
        "The benchmark results in Phase 1 and Phase 3 already reflect\n"
        "quantized model performance. The 16-17 tok/s throughput measured\n"
        "on llama3.2:3b is Q4 quantized performance — not full precision.\n"
        "Full F32 models would require 4x more memory and run significantly\n"
        "slower on CPU hardware.\n\n"
        "[bold]Production Takeaway:[/bold]\n\n"
        "For CPU-only edge deployment, always use GGUF quantized models.\n"
        "Ollama handles this automatically. For GPU deployment, F16 models\n"
        "offer a better quality-speed tradeoff than Q4.",
        title="Quantization Engineering Finding",
        border_style="green"
    ))


if __name__ == "__main__":
    main()