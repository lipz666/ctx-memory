"""Write docs/INDEX.md listing every page title."""
from pathlib import Path

docs = Path(__file__).resolve().parents[1] / "docs"
lines = ["# Index", ""]
for page in sorted(docs.glob("*.md")):
    if page.name == "INDEX.md":
        continue
    title = page.read_text().splitlines()[0].lstrip("# ").strip()
    lines.append(f"- [{title}]({page.name})")
(docs / "INDEX.md").write_text("\n".join(lines) + "\n")
