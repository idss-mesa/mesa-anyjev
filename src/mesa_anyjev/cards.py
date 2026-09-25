"""Dataset cards: the neon-avu-eval markdown grammar parsed into a frozen model.

A card (``neon-avu-eval/scripts/make_cards.py``) is::

    # Dataset: <productCode> / <table>
    Product: <code> — <title>. <description>
    Source: ...
    Sites: <CODE> (<name>, <state>, domain <Dxx> <domain name>; <habitat>) and <CODE> (...).
    Rows: <n>; months covered: <from> to <to> (<k> distinct months).

    ## Columns (name | NEON description | type | unit | profile)
    - <name> | <description> | <dtype> | <unit or -> | <profile>

Only the standard library is used, so the parser is usable from the bench and from SQL-side
state builders that must match ``states.py`` byte for byte.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_HEADER = re.compile(r"^#\s*Dataset:\s*(?P<product>\S+)\s*/\s*(?P<table>\S+)\s*$")
_PRODUCT = re.compile(r"^Product:\s*(?P<code>\S+)\s*[—-]+\s*(?P<title>[^.]+)\.?\s*(?P<desc>.*)$")
_ROWS = re.compile(r"^Rows:\s*(?P<rows>\d+);\s*months covered:\s*(?P<a>\S+)\s+to\s+(?P<b>\S+)")
_SITE = re.compile(
    r"(?P<code>[A-Z]{4})\s*\((?P<name>[^,]+),\s*(?P<state>[^,]+),\s*domain\s+(?P<domain>D\d{2})"
    r"\s*(?P<dname>[^;]*);\s*(?P<habitat>[^)]*)\)"
)
_COLUMN = re.compile(
    r"^-\s*(?P<name>[^|]+)\|(?P<desc>[^|]*)\|(?P<dtype>[^|]*)\|(?P<unit>[^|]*)\|(?P<profile>.*)$"
)

IDENTIFIER_SUFFIXES = ("id", "uid", "code", "identifiedby", "recordedby", "measuredby")


class SiteInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    name: str
    state: str
    domain: str
    domain_name: str
    habitat: str


class ColumnInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    dtype: str
    unit: str  # '' when the card shows '-'
    profile: str


class DatasetCard(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    product_code: str
    table: str
    product_title: str
    product_description: str
    source: str
    sites: list[SiteInfo] = Field(default_factory=list)
    rows: int
    months_from: str
    months_to: str
    columns: list[ColumnInfo] = Field(default_factory=list)
    sha256: str

    @property
    def name(self) -> str:
        return f"{self.product_code}.{self.table}"

    def column(self, name: str) -> ColumnInfo:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)


def card_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_card(text: str) -> DatasetCard:
    product_code = table = ""
    title = desc = source = ""
    rows = 0
    months_from = months_to = ""
    sites: list[SiteInfo] = []
    columns: list[ColumnInfo] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if m := _HEADER.match(line):
            product_code, table = m["product"], m["table"]
        elif m := _PRODUCT.match(line):
            title, desc = m["title"].strip(), m["desc"].strip()
        elif line.startswith("Source:"):
            source = line[len("Source:") :].strip()
        elif line.startswith("Sites:"):
            sites = [
                SiteInfo(
                    code=s["code"],
                    name=s["name"].strip(),
                    state=s["state"].strip(),
                    domain=s["domain"],
                    domain_name=s["dname"].strip(),
                    habitat=s["habitat"].strip(),
                )
                for s in _SITE.finditer(line)
            ]
        elif m := _ROWS.match(line):
            rows, months_from, months_to = int(m["rows"]), m["a"], m["b"]
        elif m := _COLUMN.match(line):
            unit = m["unit"].strip()
            columns.append(
                ColumnInfo(
                    name=m["name"].strip(),
                    description=m["desc"].strip(),
                    dtype=m["dtype"].strip(),
                    unit="" if unit == "-" else unit,
                    profile=m["profile"].strip(),
                )
            )
    if not product_code or not table:
        raise ValueError("not a dataset card: missing '# Dataset: <product> / <table>' header")
    return DatasetCard(
        product_code=product_code,
        table=table,
        product_title=title,
        product_description=desc,
        source=source,
        sites=sites,
        rows=rows,
        months_from=months_from,
        months_to=months_to,
        columns=columns,
        sha256=card_sha256(text),
    )


def load_card(path: str | Path) -> DatasetCard:
    return parse_card(Path(path).read_text(encoding="utf-8"))


def is_identifier(col: ColumnInfo) -> bool:
    """The deterministic pre-filter: identifiers, bookkeeping and timestamps are never
    annotated (recorded as a ``rule`` decision, never a model call)."""
    name = col.name.lower()
    if name in {"uid", "remarks", "publicationdate", "release"}:
        return True
    if name.endswith(IDENTIFIER_SUFFIXES) and name not in {"taxonid"}:
        return True
    if col.dtype.lower() in {"datetime", "date"}:
        return True
    profile = col.profile.lower()
    return profile.startswith("all blank") or "not profiled" in profile
