"""Runtime settings shared by jobs, the app and the registered agent."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

_IDENT = re.compile(r"^[A-Za-z0-9_\-]+$")

DEFAULT_LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"


@dataclass(frozen=True)
class Settings:
    catalog: str = "workspace"
    schema: str = "chiro"
    llm_endpoint: str = DEFAULT_LLM_ENDPOINT
    clinic_name: str = "Northside Chiropractic"
    # Where the real clinic tables (leads, patients, visits, appointments, providers, ...) live.
    # The landing/operational tables the rest of the repo reads are projected from here.
    source_catalog: str = "workspace"
    source_schema: str = "chiro_hackathon"

    def __post_init__(self) -> None:
        for name in ("catalog", "schema", "source_catalog", "source_schema"):
            if not _IDENT.match(getattr(self, name)):
                raise ValueError(f"Invalid {name}: {getattr(self, name)!r}")

    def table(self, name: str) -> str:
        if not _IDENT.match(name):
            raise ValueError(f"Invalid table name: {name!r}")
        return f"`{self.catalog}`.`{self.schema}`.`{name}`"

    def source_table(self, name: str) -> str:
        if not _IDENT.match(name):
            raise ValueError(f"Invalid source table name: {name!r}")
        return f"`{self.source_catalog}`.`{self.source_schema}`.`{name}`"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            catalog=os.getenv("CHIRO_CATALOG", cls.catalog),
            schema=os.getenv("CHIRO_SCHEMA", cls.schema),
            llm_endpoint=os.getenv("LLM_ENDPOINT", cls.llm_endpoint),
            clinic_name=os.getenv("CLINIC_NAME", cls.clinic_name),
            source_catalog=os.getenv("CHIRO_SOURCE_CATALOG", cls.source_catalog),
            source_schema=os.getenv("CHIRO_SOURCE_SCHEMA", cls.source_schema),
        )

    @classmethod
    def from_widgets(cls, dbutils) -> "Settings":
        """Read job parameters / notebook widgets (defaults used for interactive runs)."""
        defaults = cls()

        def get(name: str, default: str) -> str:
            dbutils.widgets.text(name, default)
            return dbutils.widgets.get(name) or default

        return cls(
            catalog=get("catalog", defaults.catalog),
            schema=get("schema", defaults.schema),
            llm_endpoint=get("llm_endpoint", defaults.llm_endpoint),
            clinic_name=get("clinic_name", defaults.clinic_name),
            source_catalog=get("source_catalog", defaults.source_catalog),
            source_schema=get("source_schema", defaults.source_schema),
        )
