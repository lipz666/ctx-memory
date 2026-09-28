import csv


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def clean(rows):
    out = []
    for row in rows:
        if not row.get("sku"):
            continue
        out.append({"sku": row["sku"].strip().upper(), "qty": int(row.get("qty") or 0)})
    return out
