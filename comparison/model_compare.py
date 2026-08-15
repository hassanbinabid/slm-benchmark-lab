"""
Phase 3 - Model Comparison Study
==================================
Benchmarks llama3.2:3b, phi4-mini, and mistral:7b on 40 standardised prompts
across 5 task categories. Records speed, latency and raw outputs for quality scoring.
Run: python comparison/model_compare.py
"""

import json
import csv
import time
import statistics
from datetime import datetime
from pathlib import Path

import ollama
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
console = Console()

MODELS = [
    "llama3.2:3b",
    "phi4-mini",
    "mistral:7b",
]

# ── 40 Prompts across 5 categories (8 each) ───────────────────────────────────
PROMPTS = [

    # Category 1 — Factual Recall
    {"id": 1,  "category": "factual", "prompt": "What is the speed of light in a vacuum?"},
    {"id": 2,  "category": "factual", "prompt": "Who wrote the theory of general relativity?"},
    {"id": 3,  "category": "factual", "prompt": "What does CPU stand for and what does it do?"},
    {"id": 4,  "category": "factual", "prompt": "What is the capital city of Australia?"},
    {"id": 5,  "category": "factual", "prompt": "What is the difference between RAM and ROM?"},
    {"id": 6,  "category": "factual", "prompt": "In what year did the World Wide Web become publicly available?"},
    {"id": 7,  "category": "factual", "prompt": "What is the chemical formula for water and why does it have that structure?"},
    {"id": 8,  "category": "factual", "prompt": "What is the difference between a compiler and an interpreter?"},

    # Category 2 — Reasoning
    {"id": 9,  "category": "reasoning", "prompt": "If all A are B, and all B are C, are all A necessarily C? Explain why."},
    {"id": 10, "category": "reasoning", "prompt": "A bat and ball cost £1.10 total. The bat costs £1 more than the ball. How much does the ball cost?"},
    {"id": 11, "category": "reasoning", "prompt": "You have 3 boxes: one has apples, one has oranges, one has both. All labels are wrong. You can pick one fruit from one box. How do you label all boxes correctly?"},
    {"id": 12, "category": "reasoning", "prompt": "Why is it hotter at the equator than at the poles? Explain the underlying reason."},
    {"id": 13, "category": "reasoning", "prompt": "If you overtake the person in second place in a race, what position are you in now?"},
    {"id": 14, "category": "reasoning", "prompt": "A doctor gives you 3 pills and says take one every 30 minutes. How long until all pills are taken?"},
    {"id": 15, "category": "reasoning", "prompt": "Explain why manhole covers are round rather than square."},
    {"id": 16, "category": "reasoning", "prompt": "You have two ropes, each takes exactly 1 hour to burn but burns unevenly. How do you measure 45 minutes?"},

    # Category 3 — Code Generation
    {"id": 17, "category": "code", "prompt": "Write a Python function that checks if a string is a palindrome."},
    {"id": 18, "category": "code", "prompt": "Write a Python function that returns the nth Fibonacci number using recursion."},
    {"id": 19, "category": "code", "prompt": "Write a Python function that flattens a nested list of arbitrary depth."},
    {"id": 20, "category": "code", "prompt": "Write a Python class for a simple stack with push, pop, and peek methods."},
    {"id": 21, "category": "code", "prompt": "Write a Python function that counts word frequencies in a string and returns a sorted dictionary."},
    {"id": 22, "category": "code", "prompt": "Write a Python decorator that measures and prints the execution time of any function."},
    {"id": 23, "category": "code", "prompt": "Write a Python function that implements bubble sort and explain each step in comments."},
    {"id": 24, "category": "code", "prompt": "Write a Python context manager that logs when a code block starts and ends."},

    # Category 4 — Creative Writing
    {"id": 25, "category": "creative", "prompt": "Write a 3 sentence story about a robot who discovers it can dream."},
    {"id": 26, "category": "creative", "prompt": "Describe a sunset on Mars in 4 sentences as if you were there."},
    {"id": 27, "category": "creative", "prompt": "Write a short poem about the feeling of debugging code at midnight."},
    {"id": 28, "category": "creative", "prompt": "Write a 3 sentence product description for an invisible umbrella."},
    {"id": 29, "category": "creative", "prompt": "Describe the taste of coffee to someone who has never tasted it, in 3 sentences."},
    {"id": 30, "category": "creative", "prompt": "Write a 4 sentence nature scene that uses only words of one syllable."},
    {"id": 31, "category": "creative", "prompt": "Write a motivational message from a senior developer to a junior developer in 3 sentences."},
    {"id": 32, "category": "creative", "prompt": "Describe what it feels like to understand a difficult concept for the first time, in 3 sentences."},

    # Category 5 — Instruction Following
    {"id": 33, "category": "instruction", "prompt": "List exactly 3 advantages and exactly 3 disadvantages of using Python for machine learning."},
    {"id": 34, "category": "instruction", "prompt": "Explain recursion in exactly 2 sentences. No more, no less."},
    {"id": 35, "category": "instruction", "prompt": "Give me exactly 5 tips for writing clean code. Number them 1 to 5."},
    {"id": 36, "category": "instruction", "prompt": "Describe the difference between SQL and NoSQL in exactly 3 bullet points."},
    {"id": 37, "category": "instruction", "prompt": "Write a haiku about artificial intelligence. A haiku is 3 lines: 5 syllables, 7 syllables, 5 syllables."},
    {"id": 38, "category": "instruction", "prompt": "Summarise what an API is in exactly one sentence of no more than 20 words."},
    {"id": 39, "category": "instruction", "prompt": "List the planets of the solar system in order from the sun. Number each one."},
    {"id": 40, "category": "instruction", "prompt": "Explain what Git is in exactly 2 sentences. First sentence for a 10 year old, second for a developer."},
]

# ── Single prompt benchmark ───────────────────────────────────────────────────
def benchmark_prompt(model: str, prompt: str) -> dict:
    """
    Run a single prompt against a model and capture timing + raw output.
    We capture the raw response so we can score quality manually later.
    """
    import httpx

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.0,
            "seed": 42,
        },
    }

    first_token_time = None
    output_tokens = 0
    full_response = ""
    start_time = time.perf_counter()

    try:
        with httpx.stream("POST", url, json=payload, timeout=180) as response:
            for line in response.iter_lines():
                if not line:
                    continue

                chunk = json.loads(line)

                # Capture first token time
                if first_token_time is None and chunk.get("response"):
                    first_token_time = time.perf_counter()

                # Accumulate full response text
                full_response += chunk.get("response", "")

                if chunk.get("done"):
                    output_tokens = chunk.get("eval_count", 0)
                    break

        end_time = time.perf_counter()

        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
        total_latency_s = end_time - start_time
        tokens_per_second = output_tokens / total_latency_s if total_latency_s > 0 else 0

        return {
            "status": "success",
            "ttft_ms": round(ttft_ms, 2),
            "total_latency_s": round(total_latency_s, 3),
            "output_tokens": output_tokens,
            "tokens_per_second": round(tokens_per_second, 2),
            "word_count": len(full_response.split()),
            "response": full_response.strip(),
        }

    except Exception as e:
        return {
            "status": "error",
            "ttft_ms": 0,
            "total_latency_s": 0,
            "output_tokens": 0,
            "tokens_per_second": 0,
            "word_count": 0,
            "response": f"ERROR: {str(e)}",
        }

# ── Main comparison runner ────────────────────────────────────────────────────
def run_comparison():
    """
    Run all models against all prompts and save results to CSV.
    This will take 20-40 minutes depending on your hardware.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"model_comparison_{timestamp}.csv"

    fieldnames = [
        "model", "prompt_id", "category", "prompt",
        "status", "ttft_ms", "total_latency_s",
        "output_tokens", "tokens_per_second", "word_count",
        "response", "quality_score"
    ]

    console.print(Panel(
        f"[bold]Models:[/bold] {', '.join(MODELS)}\n"
        f"[bold]Prompts:[/bold] {len(PROMPTS)} across 5 categories\n"
        f"[bold]Total runs:[/bold] {len(MODELS) * len(PROMPTS)}\n"
        f"[bold]Output:[/bold] {output_file}",
        title="Phase 3 - Model Comparison Study",
        border_style="blue",
    ))

    all_results = []

    for model in MODELS:
        console.print(f"\n[bold yellow]── Model: {model} ──[/bold yellow]")

        model_results = []

        for item in PROMPTS:
            console.print(
                f"  [dim]Prompt {item['id']}/40 "
                f"[{item['category']}] "
                f"{item['prompt'][:50]}...[/dim]",
                end="\r"
            )

            result = benchmark_prompt(model, item["prompt"])

            row = {
                "model": model,
                "prompt_id": item["id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "status": result["status"],
                "ttft_ms": result["ttft_ms"],
                "total_latency_s": result["total_latency_s"],
                "output_tokens": result["output_tokens"],
                "tokens_per_second": result["tokens_per_second"],
                "word_count": result["word_count"],
                "response": result["response"],
                "quality_score": "",  # You fill this in manually after the run
            }

            model_results.append(row)
            all_results.append(row)

            status_icon = "[green]✓[/green]" if result["status"] == "success" else "[red]✗[/red]"
            console.print(
                f"  {status_icon} [{item['category']}] "
                f"Prompt {item['id']:02d} — "
                f"{result['tokens_per_second']} tok/s — "
                f"{result['word_count']} words"
            )

        # Per-model summary after each model finishes
        successful = [r for r in model_results if r["status"] == "success"]
        if successful:
            avg_toks = round(statistics.mean(r["tokens_per_second"] for r in successful), 2)
            avg_ttft = round(statistics.mean(r["ttft_ms"] for r in successful), 1)
            avg_words = round(statistics.mean(r["word_count"] for r in successful), 1)

            console.print(Panel(
                f"[bold]Prompts completed:[/bold] {len(successful)}/40\n"
                f"[green]Avg tok/s:[/green]      {avg_toks}\n"
                f"[yellow]Avg TTFT:[/yellow]       {avg_ttft} ms\n"
                f"[cyan]Avg word count:[/cyan]  {avg_words}",
                title=f"Summary — {model}",
                border_style="green",
            ))

    # Save everything to CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    console.print(f"\n[green]✓ All results saved to {output_file}[/green]")
    return all_results, output_file


# ── Speed comparison table ────────────────────────────────────────────────────
def print_speed_comparison(results: list[dict]):
    """
    Print a side by side speed comparison across all three models.
    """
    table = Table(
        title="Speed Comparison — All Models",
        show_lines=True
    )

    table.add_column("Model", style="cyan", width=18)
    table.add_column("Avg Tok/s", justify="right", style="green")
    table.add_column("Avg TTFT (ms)", justify="right", style="yellow")
    table.add_column("Avg Latency (s)", justify="right", style="yellow")
    table.add_column("Avg Word Count", justify="right", style="blue")

    for model in MODELS:
        model_rows = [r for r in results if r["model"] == model and r["status"] == "success"]
        if not model_rows:
            continue

        avg_toks    = round(statistics.mean(r["tokens_per_second"] for r in model_rows), 2)
        avg_ttft    = round(statistics.mean(r["ttft_ms"] for r in model_rows), 1)
        avg_latency = round(statistics.mean(r["total_latency_s"] for r in model_rows), 2)
        avg_words   = round(statistics.mean(r["word_count"] for r in model_rows), 1)

        table.add_row(model, str(avg_toks), str(avg_ttft), str(avg_latency), str(avg_words))

    console.print(table)


# ── Category breakdown table ──────────────────────────────────────────────────
def print_category_breakdown(results: list[dict]):
    """
    Show average tokens/sec per model per category.
    Helps identify where each model is strongest and weakest.
    """
    categories = ["factual", "reasoning", "code", "creative", "instruction"]

    table = Table(
        title="Tokens/sec by Category",
        show_lines=True
    )

    table.add_column("Category", style="cyan", width=14)
    for model in MODELS:
        table.add_column(model, justify="right", style="green")

    for category in categories:
        row = [category]
        for model in MODELS:
            cat_rows = [
                r for r in results
                if r["model"] == model
                and r["category"] == category
                and r["status"] == "success"
            ]
            if cat_rows:
                avg = round(statistics.mean(r["tokens_per_second"] for r in cat_rows), 2)
                row.append(str(avg))
            else:
                row.append("—")
        table.add_row(*row)

    console.print(table)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    results, output_file = run_comparison()
    print_speed_comparison(results)
    print_category_breakdown(results)
    console.print(Panel(
        f"[bold]Next step:[/bold] Open {output_file} in Excel or VS Code.\n"
        f"Fill in the [bold yellow]quality_score[/bold yellow] column for each row:\n\n"
        f"  [green]3[/green] = Correct, complete, well structured\n"
        f"  [yellow]2[/yellow] = Correct but incomplete or slightly off\n"
        f"  [red]1[/red] = Wrong, missing the point, or garbled\n\n"
        f"Save the file, then we will generate the final report.",
        title="Quality Scoring Instructions",
        border_style="yellow",
    ))


if __name__ == "__main__":
    main()