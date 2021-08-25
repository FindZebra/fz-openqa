import rich
from rich.console import Console
from rich.table import Table


def pprint_results(data, output):
    """Print all results as a nicely using rich.table"""
    all_preds = output.argmax(-1).tolist()
    targets = data["answer.target"]
    accuracy = sum([int(x == y) for x, y in zip(all_preds, targets)]) / len(
        targets
    )

    table = Table(
        title=f"Results (Accuracy=[magenta bold]{100 * accuracy:.2f}%[/magenta bold])",
        show_lines=True,
    )
    table.add_column("Question", justify="left", style="cyan", no_wrap=False)
    table.add_column(
        "Document", justify="left", style="magenta", no_wrap=False
    )
    for idx in range(len(data["answer_choices"][0])):
        table.add_column(f"Answer {idx}", justify="center", style="white")
    for idx in range(len(data["question"])):
        row = {k: data[k][idx] for k in data.keys()}
        probs = output[idx].softmax(-1)
        pred = probs.argmax(-1)
        table.add_row(
            row["question"],
            row["document"],
            *(
                format_ans_prob(k, a, row["answer.target"], pred, p)
                for k, (a, p) in enumerate(zip(row["answer_choices"], probs))
            ),
        )
    console = Console()
    console.print(table)

    rich.print(f"Accuracy=[magenta bold]{100 * accuracy:.2f}%[/magenta bold]")


def format_ans_prob(k, txt, adx, pred, p):
    """format answer using rich.style"""
    u = f"{txt}"
    if adx == k:
        u = f"[bold underline]{u}[/underline bold]"

    u += f"\n{100 * p:.2f}%"

    if k == adx and adx == pred:
        u = f"[green]{u}[/green]"
    elif k == pred:
        u = f"[red]{u}[/red]"

    return u
