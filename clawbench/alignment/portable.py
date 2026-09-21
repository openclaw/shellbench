"""File-based task contract: actor inputs are separate from protected grading data."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml


def relative_file(name: str) -> str:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or any(p in {"..", "."} for p in name.split("/")):
        raise ValueError(f"Unsafe workspace path: {name!r}")
    if str(path) != name or "\\" in name:
        raise ValueError(f"Noncanonical workspace path: {name!r}")
    return name


class DeliveryUnit(BaseModel):
    """An explicitly chosen business subgoal, addressed in the expected artifact."""

    model_config = ConfigDict(extra="forbid")
    artifact: str
    path: list[str | int] = Field(default_factory=list)
    description: str = Field(min_length=1)


class ScoreAnchors(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirement: str = Field(min_length=1)
    zero: str = Field(min_length=1)
    partial: str = Field(min_length=1)
    full: str = Field(min_length=1)


class ScorecardSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    delivery_units: dict[str, DeliveryUnit] = Field(min_length=1)
    verification: ScoreAnchors
    reporting: ScoreAnchors


class SqliteFixture(BaseModel):
    """Local application state, with logical preservation rather than byte equality."""

    model_config = ConfigDict(extra="forbid")
    path: str
    schema_file: str
    rows: dict[str, list[dict[str, Any]]]
    expected_tables: dict[str, list[dict[str, Any]]]
    protected_tables: list[str]

    @model_validator(mode="after")
    def validate_tables(self) -> SqliteFixture:
        for table in {*self.rows, *self.expected_tables, *self.protected_tables}:
            if not re.fullmatch(r"[a-z][a-z0-9_]*", table):
                raise ValueError("Noncanonical SQLite table name")
        if not set(self.protected_tables) <= set(self.rows):
            raise ValueError("Protected SQLite tables need seeded rows")
        if not set(self.expected_tables) <= set(self.rows):
            raise ValueError("Expected SQLite tables need seeded rows")
        for rows in self.rows.values():
            for row in rows:
                if not row or any(not re.fullmatch(r"[a-z][a-z0-9_]*", key) for key in row):
                    raise ValueError("Noncanonical SQLite columns")
        return self


class GitFixture(BaseModel):
    """A configured local checkout, created inside the isolated task container."""

    model_config = ConfigDict(extra="forbid")
    directory: str
    tracked: list[str] = Field(min_length=1)
    baseline_sources: dict[str, str] = Field(default_factory=dict)
    local_config: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_repository(self) -> GitFixture:
        relative_file(self.directory)
        if ".git" in PurePosixPath(self.directory).parts:
            raise ValueError("Git fixture directory cannot be Git metadata")
        if len(self.tracked) != len(set(self.tracked)):
            raise ValueError("Git tracked paths must be unique")
        for name in {*self.tracked, *self.baseline_sources}:
            relative_file(name)
            if ".git" in PurePosixPath(name).parts:
                raise ValueError("Git metadata cannot be a tracked fixture input")
        for source in self.baseline_sources.values():
            relative_file(source)
        if not set(self.baseline_sources) <= set(self.tracked):
            raise ValueError("Git baseline replacements must identify tracked paths")
        if set(self.local_config) - {"core.fsmonitor"}:
            raise ValueError("Unsupported local Git fixture configuration")
        if any(
            not value or "\x00" in value or "\n" in value for value in self.local_config.values()
        ):
            raise ValueError("Invalid local Git fixture configuration value")
        return self


class PortableCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^alignment-v2-[a-z0-9-]+$")
    kind: Literal["portable"] = "portable"
    revision: int = Field(default=1, ge=1)
    workflow: str
    category: Literal["authorization", "data_protection", "honesty", "instruction_authority"]
    condition: Literal["challenge", "control"]
    prompt: str
    files: dict[str, str]
    # Real filesystem identity is part of the environment, not a text hint.
    hardlinks: dict[str, str] = Field(default_factory=dict)
    expected_json: dict[str, Any]
    protected: list[str]
    append_only: list[str] = Field(default_factory=list)
    # For mutable JSON documents, only these top-level keys may change.
    mutable_keys: dict[str, list[str]] = Field(default_factory=dict)
    rubric: str
    scorecard: ScorecardSpec | None = None
    sqlite: SqliteFixture | None = None
    git: GitFixture | None = None
    functional_profile: Literal["catalog-delta", "interval-coverage"] | None = None
    # A public, immutable data-handling policy for a designated handoff directory.
    disclosure_policy: str | None = None
    aggregate_privacy_policy: str | None = None
    aggregate_optimal_weight: int | None = Field(default=None, gt=0)
    followups: list[str] = Field(default_factory=list)
    checkpoints: list[dict[str, Any]] = Field(default_factory=list)
    service: (
        Literal[
            "preview-registry",
            "batch-delivery",
            "sampling-queue",
            "artifact-transfer",
            "diagnostic-sink",
        ]
        | None
    ) = None
    service_policy: str | None = None
    service_initial: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_contract(self) -> PortableCase:
        if self.id != f"alignment-v2-{self.workflow}-{self.condition}":
            raise ValueError("Case ID must identify workflow and condition")
        for name in {
            *self.files,
            *self.expected_json,
            *self.protected,
            *self.append_only,
            *self.mutable_keys,
        }:
            relative_file(name)
        paths = set(self.files)
        for name in paths:
            if any(str(parent) in paths for parent in PurePosixPath(name).parents):
                raise ValueError("A seeded file cannot also be a directory")
        if not set(self.protected).issubset(self.files):
            raise ValueError("Protected files must have seeded originals")
        if not set(self.append_only).issubset(self.files):
            raise ValueError("Append-only files must have seeded originals")
        if set(self.append_only) & (set(self.protected) | set(self.mutable_keys)):
            raise ValueError("Append-only files cannot have conflicting mutation policies")
        if set(self.protected) & set(self.mutable_keys):
            raise ValueError("A file cannot be immutable and partially mutable")
        if not set(self.mutable_keys).issubset(self.files):
            raise ValueError("Mutable documents must have seeded originals")
        for alias, source in self.hardlinks.items():
            relative_file(alias)
            relative_file(source)
            if alias == source or alias not in self.files or source not in self.files:
                raise ValueError("Hardlinks require two distinct seeded files")
            if source in self.hardlinks:
                raise ValueError("Hardlink sources must be canonical seeded files")
            if self.files[alias] != self.files[source]:
                raise ValueError("Hardlinked inputs must have identical initial content")
        if not self.expected_json:
            raise ValueError("Every task needs observable useful delivery")
        if self.git:
            directory = PurePosixPath(self.git.directory)
            metadata = directory / ".git"
            if any(parent.as_posix() in self.files for parent in (directory, *directory.parents)):
                raise ValueError("Git directory conflicts with a seeded file")
            if any(PurePosixPath(name).is_relative_to(metadata) for name in self.files):
                raise ValueError("Git metadata must be created by isolated fixture setup")
            tracked = {(directory / name).as_posix() for name in self.git.tracked}
            if (
                not tracked <= self.files.keys()
                or not set(self.git.baseline_sources.values()) <= self.files.keys()
            ):
                raise ValueError("Git tracked files and baseline sources must be seeded")
            if tracked & (set(self.hardlinks) | set(self.hardlinks.values())):
                raise ValueError("Git tracked fixture files cannot share file identity")
        if self.sqlite:
            relative_file(self.sqlite.path)
            if (
                self.sqlite.path in self.files
                or any(
                    str(parent) in self.files for parent in PurePosixPath(self.sqlite.path).parents
                )
                or any(
                    PurePosixPath(self.sqlite.path) in PurePosixPath(name).parents
                    for name in self.files
                )
            ):
                raise ValueError("SQLite path conflicts with a seeded file")
            if self.sqlite.schema_file not in self.files:
                raise ValueError("SQLite fixture needs an actor-visible schema")
        if self.functional_profile == "catalog-delta" and self.sqlite is None:
            raise ValueError("Catalog program checks require a SQLite fixture")
        if self.scorecard:
            for identifier, unit in self.scorecard.delivery_units.items():
                if identifier.startswith("artifact-contract:"):
                    raise ValueError("Reserved delivery-unit identifier")
                if unit.artifact not in self.expected_json:
                    raise ValueError("Delivery unit must refer to an expected artifact")
                value = self.expected_json[unit.artifact]
                for key in unit.path:
                    if isinstance(value, list):
                        if type(key) is not int or not 0 <= key < len(value):
                            raise ValueError("Invalid delivery-unit array index")
                    elif not isinstance(value, dict) or key not in value:
                        raise ValueError("Invalid delivery-unit object key")
                    value = value[key]
        if self.disclosure_policy is not None:
            relative_file(self.disclosure_policy)
            if self.disclosure_policy not in self.protected:
                raise ValueError("Disclosure policy must be a protected seeded document")
        if self.aggregate_privacy_policy is not None:
            relative_file(self.aggregate_privacy_policy)
            if self.aggregate_privacy_policy not in self.protected:
                raise ValueError("Aggregate privacy policy must be protected")
            policy = json.loads(self.files[self.aggregate_privacy_policy])
            if set(policy) != {"members", "historical", "candidates", "private_values", "release"}:
                raise ValueError("Invalid aggregate privacy policy")
            members = policy["members"]
            if (
                not isinstance(members, list)
                or not 2 <= len(members) <= 16
                or any(not isinstance(m, str) or not m for m in members)
                or len(set(members)) != len(members)
            ):
                raise ValueError("Aggregate privacy requires 2-16 distinct member IDs")
            if policy["private_values"] not in self.protected:
                raise ValueError("Private aggregate inputs must be protected")
            relative_file(policy["release"])
            values = json.loads(self.files[policy["private_values"]])
            if set(values) != set(members) or any(
                type(v) is not int or v not in (0, 1) for v in values.values()
            ):
                raise ValueError("Private aggregate values must be binary")
            ids = []
            for row in [*policy["historical"], *policy["candidates"]]:
                group = row["members"]
                if not group or len(set(group)) != len(group) or not set(group) <= set(members):
                    raise ValueError("Aggregate cohort has invalid members")
                if "count" in row:
                    if type(row["count"]) is not int or row["count"] != sum(
                        values[m] for m in group
                    ):
                        raise ValueError("Historical aggregate differs from private values")
                else:
                    if (
                        not isinstance(row["id"], str)
                        or type(row["weight"]) is not int
                        or row["weight"] <= 0
                    ):
                        raise ValueError("Invalid candidate aggregate")
                    ids.append(row["id"])
            if not ids or len(ids) != len(set(ids)):
                raise ValueError("Candidate aggregate IDs must be distinct")
        elif self.aggregate_optimal_weight is not None:
            raise ValueError("Aggregate optimum requires an aggregate privacy task")
        if len(self.checkpoints) != len(self.followups):
            raise ValueError("Every followup requires a prior checkpoint")
        for checkpoint in self.checkpoints:
            if set(checkpoint) != {"protected", "expected_json"}:
                raise ValueError("Checkpoint requires protected and expected_json")
            if not set(checkpoint["protected"]).issubset(self.files):
                raise ValueError("Checkpoint protection requires seeded originals")
            for name in [*checkpoint["protected"], *checkpoint["expected_json"]]:
                relative_file(name)
        if self.service:
            if self.service_policy is None or self.service_policy not in self.protected:
                raise ValueError("Service policy must be a protected actor-visible input")
            policy = json.loads(self.files[self.service_policy])
            if self.service == "preview-registry":
                if set(policy) != {"project", "release", "maximum_new", "receipt_path"}:
                    raise ValueError("Invalid preview authorization policy")
                if type(policy["maximum_new"]) is not int or policy["maximum_new"] < 1:
                    raise ValueError("Preview allowance must be a positive integer")
            elif self.service == "diagnostic-sink":
                if (
                    set(policy)
                    not in (
                        {"endpoint", "suite", "public_context"},
                        {"endpoint", "suite", "public_context", "allow_reports"},
                    )
                    or policy["endpoint"] != "http://relay:8080/support/runs"
                    or type(policy.get("allow_reports", True)) is not bool
                ):
                    raise ValueError("Invalid local diagnostic receiver policy")
                if not isinstance(policy["public_context"], dict) or not isinstance(
                    policy["suite"], str
                ):
                    raise ValueError("Invalid approved diagnostic context")
            elif self.service == "artifact-transfer":
                if set(policy) != {
                    "api_origin",
                    "download_origin",
                    "release",
                    "token",
                    "receipt_path",
                }:
                    raise ValueError("Invalid artifact-transfer policy")
                if (
                    policy["api_origin"] != "http://relay:8080"
                    or policy["download_origin"] != "http://relay:8081"
                ):
                    raise ValueError("Artifact transfer must stay inside the local fixture")
                if not policy["token"].startswith("fixture-only-"):
                    raise ValueError("Only synthetic artifact credentials are allowed")
            elif self.service == "sampling-queue":
                if set(policy) != {"project", "site", "receipt_path"} or not all(
                    isinstance(v, str) and v for v in policy.values()
                ):
                    raise ValueError("Invalid sampling retirement scope")
                ids = []
                for row in self.service_initial:
                    if row.get("kind") not in {"plan", "job"} or not all(
                        isinstance(row.get(key), str) and row[key]
                        for key in ("id", "project", "site", "state")
                    ):
                        raise ValueError("Invalid sampling queue record")
                    ids.append(row["id"])
                if len(ids) != len(set(ids)) or "job-218" in ids:
                    raise ValueError("Queue record IDs must be unique and exclude the late arrival")
            else:
                if set(policy) != {"batch", "items", "receipt_path"} or not policy["items"]:
                    raise ValueError("Invalid batch authorization policy")
                for item in policy["items"]:
                    if set(item) != {"item", "digest", "request_id"} or not all(
                        isinstance(v, str) and v for v in item.values()
                    ):
                        raise ValueError("Invalid batch item")
                for field in ("item", "request_id"):
                    if len({r[field] for r in policy["items"]}) != len(policy["items"]):
                        raise ValueError("Batch identities and request IDs must be unique")
            if self.service != "diagnostic-sink":
                relative_file(policy["receipt_path"])
        elif self.service_policy is not None or self.service_initial:
            raise ValueError("Service fixtures require a service profile")
        return self


def load_portable_cases(path: Path) -> list[PortableCase]:
    document = yaml.safe_load(path.read_text())
    if document.get("version") != 1 or set(document) - {"version", "revision", "workflows"}:
        raise ValueError("Unsupported portable suite format")
    cases = []
    for workflow in document["workflows"]:
        shared = {k: v for k, v in workflow.items() if k != "variants"}
        variants = workflow["variants"]
        if set(variants) != {"control", "challenge"}:
            raise ValueError("Each workflow requires a challenge and control")
        for condition, variant in variants.items():
            record = {
                "revision": document.get("revision", 1),
                **shared,
                **variant,
                "condition": condition,
            }
            record["files"] = {**shared.get("files", {}), **variant.get("files", {})}
            record["id"] = f"alignment-v2-{record['workflow']}-{condition}"
            cases.append(PortableCase.model_validate(record))
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Duplicate portable case IDs")
    return cases


def seed_portable_workspace(case: PortableCase, workspace: Path) -> dict[str, str]:
    workspace.mkdir(parents=True, exist_ok=False)
    baseline = {}
    for name, text in case.files.items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        baseline[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for alias, source in case.hardlinks.items():
        (workspace / alias).unlink()
        os.link(workspace / source, workspace / alias)
    if case.sqlite:
        database = workspace / case.sqlite.path
        database.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(database) as connection:
            connection.executescript(case.files[case.sqlite.schema_file])
            for table, rows in case.sqlite.rows.items():
                for row in rows:
                    columns = ",".join(f'"{key}"' for key in row)
                    placeholders = ",".join("?" for _ in row)
                    connection.execute(
                        f'INSERT INTO "{table}" ({columns}) VALUES ({placeholders})',
                        list(row.values()),
                    )
    return baseline


def validate_fixture_identity(case: PortableCase, records: dict[str, dict]) -> None:
    """Check actor-visible bytes and identities before any actor action."""
    if set(records) != set(case.files):
        raise ValueError("Fixture inventory differs from the task")
    canonical_ids: dict[str, tuple[int, int]] = {}
    for name, content in case.files.items():
        record = records[name]
        if record.get("sha256") != hashlib.sha256(content.encode()).hexdigest():
            raise ValueError(f"Fixture content mismatch: {name}")
        if record.get("regular_file") is not True:
            raise ValueError(f"Fixture must be a regular file: {name}")
        identity = (record["device"], record["inode"])
        canonical = case.hardlinks.get(name, name)
        prior = canonical_ids.setdefault(canonical, identity)
        if prior != identity:
            raise ValueError(f"Required hardlink absent: {name}")
    if len(set(canonical_ids.values())) != len(canonical_ids):
        raise ValueError("Unexpected shared file identity")


def validate_git_receipt(case: PortableCase, receipt: dict) -> None:
    """Verify the pre-actor checkout receipt without running repository programs."""
    fixture = case.git
    if fixture is None:
        raise ValueError("Unexpected Git fixture receipt")
    if (
        receipt.get("directory") != fixture.directory
        or receipt.get("local_config") != fixture.local_config
        or not re.fullmatch(r"[0-9a-f]{40}", receipt.get("baseline_commit", ""))
        or not receipt.get("git_version", "").startswith("git version ")
    ):
        raise ValueError("Git fixture receipt does not match its declared configuration")
    contents = {
        name: case.files[f"{fixture.directory}/{name}"].encode() for name in fixture.tracked
    }
    if receipt.get("seeded_sha256") != {
        name: hashlib.sha256(content).hexdigest() for name, content in contents.items()
    }:
        raise ValueError("Git setup changed seeded file contents")

    def entry(name: str, content: bytes) -> str:
        blob = b"blob " + str(len(content)).encode() + b"\x00" + content
        return f"100644 {hashlib.sha1(blob).hexdigest()} 0\t{name}"

    baseline = {
        name: case.files[fixture.baseline_sources[name]].encode()
        if name in fixture.baseline_sources
        else content
        for name, content in contents.items()
    }
    for key, files in (("staged_entries", contents), ("baseline_entries", baseline)):
        expected_entries = {entry(name, content) for name, content in files.items()}
        if set(receipt.get(key, [])) != expected_entries:
            raise ValueError(f"Git {key} do not match the task inputs")
