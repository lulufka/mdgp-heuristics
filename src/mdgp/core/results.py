import csv
from pathlib import Path


def save_results_csv(results: list[dict], path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        return

    fieldnames = list(results[0].keys())

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)