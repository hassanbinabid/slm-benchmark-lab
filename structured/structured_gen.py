"""
Phase 2 - Structured Output Engineering
Pattern: Constrained prompt -> JSON parse -> Pydantic validate -> Retry once -> Fail gracefully
Run: python structured/structured_gen.py
"""

import json
import csv
import time
from datetime import datetime
from pathlib import Path

import ollama
from pydantic import BaseModel, ValidationError, field_validator
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "llama3.2:3b"
RESULTS_DIR = Path("results")
console = Console()


# ── Pydantic Schema ───────────────────────────────────────────────────────────
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
        words = v.split()
        if len(words) > 20:
            raise ValueError(f"one_line_summary too long: {len(words)} words (max 20)")
        return v


# ── Prompt Builder ────────────────────────────────────────────────────────────
def build_prompt(topic: str, previous_error: str = None) -> str:
    base_prompt = f"""You are a technical educator. Explain the concept: "{topic}"

You must respond with ONLY a raw JSON object.
No markdown, no code blocks, no extra text before or after.
Just the JSON object itself.

The JSON must have exactly these fields:
- "topic": string (the concept name)
- "difficulty": string (must be exactly one of: easy, intermediate, advanced)
- "explanation": string (plain English, max 80 words)
- "has_code_example": boolean (true if your explanation mentions code, false otherwise)
- "one_line_summary": string (max 15 words summarising the concept)

Example of valid response:
{{
    "topic": "recursion",
    "difficulty": "intermediate",
    "explanation": "Recursion is when a function calls itself to solve a problem.",
    "has_code_example": false,
    "one_line_summary": "A function that calls itself to solve smaller versions of a problem."
}}"""

    if previous_error:
        base_prompt += f"""

Your previous response failed validation with this error:
{previous_error}

Please fix that specific issue and return the full JSON again."""

    return base_prompt


# ── Model Caller ──────────────────────────────────────────────────────────────
def call_model(prompt: str, temperature: float) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature, "seed": 42},
    )
    return response["message"]["content"].strip()

# ── Retry Logic ───────────────────────────────────────────────────────────────
def generate_structured(topic: str, temperature: float) -> dict:
    """
    Full pipeline:
    1. Build prompt
    2. Call model
    3. Parse JSON
    4. Validate with Pydantic
    5. If fails -> retry once with error context
    6. If fails again -> return failure record
    """

    attempt = 1
    previous_error = None

    while attempt <= 2:
        console.print(f"  [dim]Attempt {attempt} | temp={temperature}[/dim]")

        # Step 1 - build prompt (includes error context on retry)
        prompt = build_prompt(topic, previous_error)

        # Step 2 - call the model
        raw_response = call_model(prompt, temperature)

        # Step 3 - try to parse as JSON
        try:
            parsed_json = json.loads(raw_response)
        except json.JSONDecodeError as e:
            previous_error = f"Response was not valid JSON: {e}"
            console.print(f"  [red]✗ JSON parse failed: {e}[/red]")
            attempt += 1
            continue

        # Step 4 - validate with Pydantic
        try:
            validated = ConceptExplanation(**parsed_json)
            console.print(f"  [green]✓ Validated successfully on attempt {attempt}[/green]")
            return {
                "topic": topic,
                "temperature": temperature,
                "attempt": attempt,
                "status": "success",
                "difficulty": validated.difficulty,
                "has_code_example": validated.has_code_example,
                "one_line_summary": validated.one_line_summary,
                "explanation": validated.explanation,
                "error": None,
            }

        except ValidationError as e:
            previous_error = str(e.errors()[0]["msg"])
            console.print(f"  [red]✗ Validation failed: {previous_error}[/red]")
            attempt += 1

    # Step 5 - both attempts failed, return failure record
    console.print(f"  [bold red]✗ Failed after 2 attempts[/bold red]")
    return {
        "topic": topic,
        "temperature": temperature,
        "attempt": 2,
        "status": "failed",
        "difficulty": None,
        "has_code_example": None,
        "one_line_summary": None,
        "explanation": None,
        "error": previous_error,
    }

# ── Temperature Experiment ────────────────────────────────────────────────────
def run_temperature_experiment():
    """
    Run the same 5 topics at temperature 0.0 and 0.7.
    Documents how much output varies between the two settings.
    """

    topics = [
        "recursion",
        "gradient descent",
        "TCP/IP protocol",
        "binary search",
        "REST API",
    ]

    temperatures = [0.0, 0.7]
    all_results = []

    console.print(Panel(
        f"[bold]Model:[/bold] {MODEL}\n"
        f"[bold]Topics:[/bold] {len(topics)}\n"
        f"[bold]Temperatures:[/bold] {temperatures}\n"
        f"[bold]Total runs:[/bold] {len(topics) * len(temperatures)}",
        title="Phase 2 - Structured Generation + Temperature Experiment",
        border_style="blue",
    ))

    for temperature in temperatures:
        console.print(f"\n[bold yellow]── Temperature: {temperature} ──[/bold yellow]")

        for topic in topics:
            console.print(f"\n[cyan]Topic: {topic}[/cyan]")
            result = generate_structured(topic, temperature)
            all_results.append(result)

    return all_results


# ── Print Results Table ───────────────────────────────────────────────────────
def print_results(results: list[dict]):
    table = Table(
        title="Structured Generation Results",
        show_lines=True,
    )

    table.add_column("Topic", style="cyan", max_width=20)
    table.add_column("Temp", justify="center", width=5)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Attempt", justify="center", width=7)
    table.add_column("Difficulty", justify="center", width=12)
    table.add_column("Has Code", justify="center", width=8)
    table.add_column("Summary", max_width=35)

    for r in results:
        status_color = "green" if r["status"] == "success" else "red"
        table.add_row(
            r["topic"],
            str(r["temperature"]),
            f"[{status_color}]{r['status']}[/{status_color}]",
            str(r["attempt"]),
            str(r["difficulty"]),
            str(r["has_code_example"]),
            str(r["one_line_summary"]) if r["one_line_summary"] else "[red]failed[/red]",
        )

    console.print(table)


# ── Save Results ──────────────────────────────────────────────────────────────
def save_results(results: list[dict]):
    RESULTS_DIR.mkdir(exist_ok=True)
    filepath = RESULTS_DIR / "structured_results.csv"

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    console.print(f"\n[green]✓ Results saved to {filepath}[/green]")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    results = run_temperature_experiment()
    print_results(results)
    save_results(results)

    # Quick temperature comparison summary
    success_at_0 = sum(1 for r in results if r["temperature"] == 0.0 and r["status"] == "success")
    success_at_07 = sum(1 for r in results if r["temperature"] == 0.7 and r["status"] == "success")
    total_topics = len(set(r["topic"] for r in results))

    console.print(Panel(
        f"[bold]Temperature 0.0[/bold] — {success_at_0}/{total_topics} succeeded\n"
        f"[bold]Temperature 0.7[/bold] — {success_at_07}/{total_topics} succeeded\n\n"
        f"[dim]Check results/structured_results.csv for full output comparison[/dim]",
        title="Temperature Experiment Summary",
        border_style="green",
    ))


if __name__ == "__main__":
    main()