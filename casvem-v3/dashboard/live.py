"""
Live terminal dashboard — shows cache performance and exact cost savings in real time.
Opt 3: all costs computed from actual token counts, not estimates.

Run standalone: python -m dashboard.live
Or import start_dashboard() to run alongside the API server.
"""

import threading
import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config import cfg
from core.storage import get_storage


def _build_display() -> Panel:
    storage = get_storage()
    stats = storage.get_stats()

    total = stats["total_queries"]
    hits = stats["cache_hits"]
    misses = stats["cache_misses"]
    hit_rate = stats["hit_rate"]
    memory_count = stats["memory_count"]
    cache_entries = stats["cache_entries"]

    # Opt 3: exact cost from real token counts
    cold_input = stats["cold_input_tokens"]
    cold_output = stats["cold_output_tokens"]
    actual_cost = (
        cold_input / 1000 * cfg.gemini_cost_per_1k_input
        + cold_output / 1000 * cfg.gemini_cost_per_1k_output
    )
    mem0_cost = total * cfg.mem0_cost_per_query
    savings = mem0_cost - actual_cost
    projected_monthly = savings * 30 if total > 0 else 0

    # Hit rate trend indicator
    trend = "↑" if hit_rate > 0.5 else ("→" if hit_rate > 0.2 else "↓")
    hit_rate_pct = f"{hit_rate * 100:.1f}%  {trend}"

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold cyan", width=28)
    table.add_column("Value", style="white")

    table.add_row("Queries served", f"{total:,}")
    table.add_row("Cache hit rate", hit_rate_pct)
    table.add_row("  L1 + L2 hits", f"{hits:,}")
    table.add_row("  Cold misses", f"{misses:,}")
    table.add_row("─" * 28, "─" * 20)
    table.add_row("Input tokens used", f"{cold_input:,}")
    table.add_row("Output tokens used", f"{cold_output:,}")
    table.add_row("Actual Gemini cost", f"${actual_cost:.4f}")
    table.add_row("Mem0 equivalent cost", f"${mem0_cost:.4f}")
    table.add_row("Saved vs Mem0", f"[bold green]${savings:.4f}[/bold green]")
    table.add_row("Projected savings/mo", f"[bold green]${projected_monthly:.2f}[/bold green]")
    table.add_row("─" * 28, "─" * 20)
    table.add_row("Memories stored", f"{memory_count:,}")
    table.add_row("Cache entries", f"{cache_entries:,}")

    return Panel(
        table,
        title="[bold blue]CaSVeM v3 — Live Dashboard[/bold blue]",
        subtitle=f"[dim]LLM: {cfg.llm_backend}/{cfg.llm_model}  |  "
                 f"Cache: {'MLP' if cfg.use_mlp else 'LRU'}  |  "
                 f"Refresh: 2s[/dim]",
        border_style="blue",
    )


def run_dashboard():
    """Block and display live dashboard. Ctrl-C to exit."""
    console = Console()
    with Live(_build_display(), console=console, refresh_per_second=0.5) as live:
        while True:
            time.sleep(2)
            live.update(_build_display())


def start_dashboard_thread():
    """Start dashboard in a background daemon thread."""
    t = threading.Thread(target=run_dashboard, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    run_dashboard()
