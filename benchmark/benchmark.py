"""
Phase 1 — Inference Benchmarking
Measures tokens/sec, time to first token, and total latency.
Run: python benchmark/benchmark.py
"""

import json
import time
import csv
import statistics
from datetime import datetime
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# ── Config ──────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "llama3.2:3b"
RESULTS_DIR = Path("results")

PROMPTS = [
    "What is the capital of France?",
    "Define recursion in one sentence.",
    "Explain the difference between a list and a tuple in Python.",
    "What is gradient descent in machine learning?",
    "Write a Python function that checks if a number is prime.",
    "Explain how TCP/IP works in plain English.",
    "What are the main causes of the French Revolution?",
    "Write a Python function that reverses a string.",
    "Explain the CAP theorem in distributed systems.",
    "What is the difference between SQL and NoSQL databases?",
]

console = Console()


# ── Core measurement function ────────────────────────────────────────────────
def benchmark_prompt(prompt: str, index: int) -> dict:
    """
    Send a single prompt to Ollama and capture timing metrics.
    Uses streaming so we can detect the exact moment the first token arrives.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.0,  # Deterministic — same output every run
            "seed": 42,
        },
    }

    first_token_time = None
    output_tokens = 0
    start_time = time.perf_counter()

    with httpx.stream("POST", url, json=payload, timeout=120) as response:
        for line in response.iter_lines():
            if not line:
                continue

            chunk = json.loads(line)

            # Record the moment first token arrives
            if first_token_time is None and chunk.get("response"):
                first_token_time = time.perf_counter()

            if chunk.get("done"):
                output_tokens = chunk.get("eval_count", 0)
                break

    end_time = time.perf_counter()

    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
    total_latency_s = end_time - start_time
    tokens_per_second = output_tokens / total_latency_s if total_latency_s > 0 else 0

    return {
        "index": index,
        "prompt": prompt[:60],
        "ttft_ms": round(ttft_ms, 2),
        "total_latency_s": round(total_latency_s, 3),
        "output_tokens": output_tokens,
        "tokens_per_second": round(tokens_per_second, 2),
    }


# ── Save results to CSV ──────────────────────────────────────────────────────
def save_results(results: list[dict]):
    RESULTS_DIR.mkdir(exist_ok=True)
    filepath = RESULTS_DIR / "benchmark_results.csv"

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    console.print(f"\n[green]✓ Results saved to {filepath}[/green]")


# ── Print summary table ──────────────────────────────────────────────────────
def print_summary(results: list[dict]):
    table = Table(title=f"Benchmark Results — {MODEL}", show_lines=True)

    table.add_column("#", style="dim", width=3)
    table.add_column("Prompt (preview)", style="cyan", max_width=40)
    table.add_column("TTFT (ms)", justify="right", style="yellow")
    table.add_column("Latency (s)", justify="right", style="yellow")
    table.add_column("Tokens", justify="right")
    table.add_column("Tok/sec", justify="right", style="green")

    for r in results:
        table.add_row(
            str(r["index"]),
            r["prompt"],
            str(r["ttft_ms"]),
            str(r["total_latency_s"]),
            str(r["output_tokens"]),
            str(r["tokens_per_second"]),
        )

    console.print(table)

    # Aggregate stats
    ttfts = [r["ttft_ms"] for r in results]
    latencies = [r["total_latency_s"] for r in results]
    toks = [r["tokens_per_second"] for r in results]

    console.print(Panel(
        f"[bold]Model:[/bold] {MODEL}\n"
        f"[bold]Prompts run:[/bold] {len(results)}\n\n"
        f"[yellow]Avg TTFT:[/yellow]        {round(statistics.mean(ttfts), 1)} ms\n"
        f"[yellow]P95 TTFT:[/yellow]        {round(sorted(ttfts)[int(len(ttfts)*0.95)-1], 1)} ms\n\n"
        f"[yellow]Avg Latency:[/yellow]     {round(statistics.mean(latencies), 3)} s\n"
        f"[yellow]P95 Latency:[/yellow]     {round(sorted(latencies)[int(len(latencies)*0.95)-1], 3)} s\n\n"
        f"[green]Avg Tok/sec:[/green]      {round(statistics.mean(toks), 1)}\n"
        f"[green]Min Tok/sec:[/green]      {min(toks)}\n"
        f"[green]Max Tok/sec:[/green]      {max(toks)}\n",
        title="Summary",
        border_style="blue",
    ))


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    console.print(Panel(
        f"[bold]Model:[/bold] {MODEL}\n"
        f"[bold]Prompts:[/bold] {len(PROMPTS)}\n"
        f"[bold]Temperature:[/bold] 0.0 (deterministic)",
        title="Phase 1 — Inference Benchmark",
        border_style="blue",
    ))

    results = []

    for i, prompt in enumerate(PROMPTS, 1):
        console.print(f"[dim]Running prompt {i}/{len(PROMPTS)}...[/dim]", end="\r")
        result = benchmark_prompt(prompt, i)
        results.append(result)
        console.print(f"[green]✓[/green] Prompt {i} — {result['tokens_per_second']} tok/s")

    print_summary(results)
    save_results(results)


if __name__ == "__main__":
    main()