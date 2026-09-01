"""Matching a dataset's columns to a use case's declared inputs.

This is the second of the two moments a model is allowed to cost anything, and
it is deliberately the cheaper one: a list of column names against a list of
field names, once per workflow, rather than a page of accessibility tree on
every turn.

Most of it is not a language problem at all
-------------------------------------------
``Customer Email`` to ``customer_email`` is string normalisation. A date column
cannot fill a number field, which is types. A column whose every value matches
``...@...`` filling a field that was recorded with an email in it is value
shape. Four deterministic rules, run in order, settle the overwhelming majority
of real files for nothing:

1. **exact** on normalised names;
2. **similar** -- token overlap plus a character-level ratio;
3. **type** -- compatible, or it is not a candidate at all;
4. **shape** -- email, url, date, phone agreeing on both sides.

Only what survives ambiguously is worth a model call, and :func:`unresolved`
is what names that set. A file with well-chosen headers therefore maps for
zero tokens, and the model is the fallback rather than the mechanism.

Nothing here decides anything
-----------------------------
Every suggestion is confirmed by a person in the UI before it runs. That is not
politeness: an automatic mapping that is wrong and unreviewed does not fail, it
succeeds a thousand times into the wrong fields, and the first anyone hears of
it is from whoever owns the data on the other end. So this module ranks and
explains; it does not commit.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Iterable

from ingest import ColumnProfile

log = logging.getLogger(__name__)

#: Above this, a suggestion is offered pre-selected. Below it, the UI shows it
#: as a guess to check. Tuned so that an exact or near-exact name match lands
#: above and a type-only agreement does not.
CONFIDENT = 0.75

#: Below this a candidate is not worth showing at all.
FLOOR = 0.25

#: Which column kinds can fill which declared input types. The keys are
#: exactly ``InputSpec.type``'s literals, so a new field type cannot be added
#: to the schema without this refusing to guess about it.
#:
#: ``string`` accepts anything -- a form field is a string in the end -- but a
#: field declared as a number must not be offered a date column, because that
#: failure appears on row 1 and on every row after it.
_COMPATIBLE: dict[str, frozenset[str]] = {
    "string": frozenset({"text", "integer", "number", "date", "boolean", "empty"}),
    "url": frozenset({"text", "empty"}),
    "number": frozenset({"integer", "number", "empty"}),
    "boolean": frozenset({"boolean", "empty"}),
    "file": frozenset({"text", "empty"}),
}

#: Words that carry no distinguishing information in a header. Dropping them
#: is what lets "Customer Email Address" match "email".
_NOISE = frozenset(
    {"the", "a", "an", "of", "for", "to", "id", "no", "num", "value", "field", "column", "data"}
)

#: Names that mean the same thing in the two vocabularies a user is joining:
#: what their spreadsheet calls a column, and what the recording called a
#: field. Deliberately short -- this is a list of genuine synonyms, not a
#: dictionary, and every entry is one somebody has to maintain.
_SYNONYMS: dict[str, str] = {
    "e_mail": "email",
    "mail": "email",
    "email_address": "email",
    "phone_number": "phone",
    "telephone": "phone",
    "mobile": "phone",
    "cell": "phone",
    "zip": "postcode",
    "zipcode": "postcode",
    "postal_code": "postcode",
    "post_code": "postcode",
    "surname": "last_name",
    "family_name": "last_name",
    "given_name": "first_name",
    "forename": "first_name",
    "company": "organisation",
    "organization": "organisation",
    "org": "organisation",
    "url": "link",
    "website": "link",
    "web_address": "link",
    "qty": "quantity",
    "amount": "quantity",
    "dob": "date_of_birth",
    "birthday": "date_of_birth",
}

_SPLIT = re.compile(r"[^a-z0-9]+")


@dataclass(slots=True)
class Candidate:
    column: str
    score: float
    reason: str


@dataclass(slots=True)
class Suggestion:
    """One declared field, and what the dataset offers for it."""

    field: str
    column: str | None
    score: float
    reason: str
    #: Runners-up, best first, so the UI can offer a dropdown rather than a
    #: free-text box -- the user is choosing among real columns, which is the
    #: same reason the healer chooses among real page elements.
    alternatives: list[Candidate] = field(default_factory=list)

    @property
    def confident(self) -> bool:
        return self.column is not None and self.score >= CONFIDENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "column": self.column,
            "score": round(self.score, 3),
            "confident": self.confident,
            "reason": self.reason,
            "alternatives": [
                {"column": c.column, "score": round(c.score, 3), "reason": c.reason}
                for c in self.alternatives
            ],
        }


def normalise(name: str) -> str:
    """A header and a field name reduced to the same vocabulary."""
    tokens = [t for t in _SPLIT.split(name.strip().lower()) if t]
    joined = "_".join(tokens)
    return _SYNONYMS.get(joined, joined)


def _tokens(name: str) -> list[str]:
    tokens = [t for t in _SPLIT.split(normalise(name)) if t and t not in _NOISE]
    return [_SYNONYMS.get(t, t) for t in tokens] or [normalise(name)]


def _name_score(field_name: str, column_name: str) -> tuple[float, str]:
    a, b = normalise(field_name), normalise(column_name)
    if a == b:
        return 1.0, "the names match"

    at, bt = set(_tokens(field_name)), set(_tokens(column_name))
    if at and at == bt:
        return 0.95, "the names match once wording is normalised"

    overlap = len(at & bt) / len(at | bt) if at | bt else 0.0
    ratio = SequenceMatcher(None, a, b).ratio()
    # Token overlap leads: "email" inside "customer_email_address" is a
    # stronger signal than two names happening to share letters.
    blended = 0.65 * overlap + 0.35 * ratio

    if at & bt:
        shared = ", ".join(sorted(at & bt))
        return blended, f"both mention {shared}"
    if ratio > 0.8:
        return blended, "the names are nearly the same"
    return blended, "the names are unrelated"


def _compatible(field_type: str, column: ColumnProfile) -> bool:
    allowed = _COMPATIBLE.get(field_type, _COMPATIBLE["string"])
    return column.kind in allowed


def score(field_name: str, field_type: str, column: ColumnProfile) -> Candidate | None:
    """How well one column fits one field, or None if it cannot fit at all."""
    if not _compatible(field_type, column):
        return None

    value, reason = _name_score(field_name, column.name)

    # A recognised shape on both sides is worth more than the names, because a
    # column of addresses named "Contact" is still a column of addresses.
    field_shape = _shape_of_field(field_name, field_type)
    if column.shape and field_shape == column.shape:
        value = max(value, 0.8)
        reason = f"both look like {column.shape} values"
    elif column.shape and field_shape and field_shape != column.shape:
        # Actively wrong: the names may agree but the contents disagree.
        value *= 0.5
        reason = f"the column holds {column.shape} values, not {field_shape}"

    if column.is_empty:
        value *= 0.5
        reason = "the column is empty"

    return Candidate(column=column.name, score=max(0.0, min(1.0, value)), reason=reason)


def _shape_of_field(name: str, field_type: str = "string") -> str | None:
    """What a field implies it holds, from its declared type or failing that
    its name. The counterpart to the profile's shape, so the two compare.

    The declared type wins: it was reviewed by a person when the use case was
    published, whereas the name is whatever the recording happened to call it.
    """
    if field_type == "url":
        return "url"
    tokens = set(_tokens(name))
    if "email" in tokens:
        return "email"
    if "phone" in tokens:
        return "phone"
    if "link" in tokens:
        return "url"
    if tokens & {"date", "date_of_birth", "when"}:
        return "date"
    return None


def suggest(
    fields: Iterable[tuple[str, str]], columns: Iterable[ColumnProfile]
) -> list[Suggestion]:
    """Rank every column against every field.

    ``fields`` is ``(name, type)`` pairs, which is what ``UseCase.inputs``
    provides without this module having to import the schema.

    A column is offered to more than one field where it genuinely fits both;
    resolving that is the user's call, and pre-empting it by assigning columns
    greedily produced worse answers than showing the ranking honestly.
    """
    profiles = list(columns)
    suggestions: list[Suggestion] = []

    for name, field_type in fields:
        candidates = [
            candidate
            for candidate in (score(name, field_type, column) for column in profiles)
            if candidate is not None and candidate.score >= FLOOR
        ]
        candidates.sort(key=lambda c: c.score, reverse=True)

        if not candidates:
            reason = (
                "no column holds values this field can take"
                if profiles
                else "the dataset has no columns"
            )
            suggestions.append(Suggestion(field=name, column=None, score=0.0, reason=reason))
            continue

        best = candidates[0]
        suggestions.append(
            Suggestion(
                field=name,
                column=best.column,
                score=best.score,
                reason=best.reason,
                alternatives=candidates[1:4],
            )
        )
    return suggestions


def unresolved(suggestions: Iterable[Suggestion]) -> list[Suggestion]:
    """The suggestions a model could usefully improve.

    Two cases, and the second is the one that matters: nothing was found, or
    the top two candidates are close enough that the ranking is not really a
    decision. Sending the confident ones as well would cost tokens to be told
    what is already known.
    """
    ambiguous: list[Suggestion] = []
    for suggestion in suggestions:
        if suggestion.column is None:
            ambiguous.append(suggestion)
            continue
        if suggestion.score < CONFIDENT:
            ambiguous.append(suggestion)
            continue
        runner_up = suggestion.alternatives[0].score if suggestion.alternatives else 0.0
        if suggestion.score - runner_up < 0.1:
            ambiguous.append(suggestion)
    return ambiguous


def apply_choice(
    suggestions: list[Suggestion], field_name: str, column: str | None, reason: str
) -> None:
    """Record a decision made elsewhere -- by the model, or by a person.

    The chosen column must already be one of the candidates. That is the same
    invariant healing keeps about page elements, for the same reason: a name
    that came from a model and not from the data has no route into something
    that will run a thousand times.
    """
    for suggestion in suggestions:
        if suggestion.field != field_name:
            continue
        known = {suggestion.column, *(c.column for c in suggestion.alternatives)}
        if column is not None and column not in known:
            log.warning(
                "ignoring a mapping choice that names no offered column",
                extra={"field": field_name, "column": column},
            )
            return
        suggestion.column = column
        suggestion.score = 1.0 if column else 0.0
        suggestion.reason = reason
        return


def as_mapping(suggestions: Iterable[Suggestion]) -> dict[str, str]:
    """``{field: column}`` for the fields that have one."""
    return {s.field: s.column for s in suggestions if s.column}


def apply_mapping(
    rows: Iterable[dict[str, Any]], mapping: dict[str, str]
) -> list[dict[str, Any]]:
    """Rename dataset columns to declared field names.

    Everything downstream -- validation, templating, the results file -- works
    in field names, so this is the one place the two vocabularies meet. Columns
    that were not mapped are dropped rather than passed through, because an
    unmapped column reaching ``validate_rows`` reports as an unknown input and
    sends the user looking for a problem they already solved.
    """
    return [{field_name: row.get(column, "") for field_name, column in mapping.items()} for row in rows]


__all__ = [
    "CONFIDENT",
    "Candidate",
    "Suggestion",
    "apply_choice",
    "apply_mapping",
    "as_mapping",
    "normalise",
    "score",
    "suggest",
    "unresolved",
]
