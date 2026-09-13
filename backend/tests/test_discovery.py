"""Two-pass discovery: read a list page into rows, then run on those rows.

The row-driven model everything else uses assumes you already know the four
thousand account numbers. Against a vendor who will not open their back end,
you do not -- the list page *is* the index. These tests pin the pass that turns
it into one, and the join that makes its output runnable.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from credentials import generate_key
from usecase import Step, UseCase
from test_api_execute import FakeReplaySession

pytestmark = pytest.mark.anyio


# --- the schema ------------------------------------------------------------


def test_a_discovery_step_must_say_what_to_read_out_of_each_row():
    """Rows with no columns would find them and read nothing."""
    with pytest.raises(ValueError) as caught:
        Step(
            id="d1",
            action="extract_rows",
            output="accounts",
            locators=[{"strategy": "css", "selector": "table tbody tr"}],
        )
    assert "at least one column" in str(caught.value)


def test_a_discovery_step_must_name_where_its_rows_land():
    with pytest.raises(ValueError) as caught:
        Step(
            id="d1",
            action="extract_rows",
            locators=[{"strategy": "css", "selector": "tr"}],
            columns=[{"name": "id", "selector": "td"}],
        )
    assert "output name" in str(caught.value)


def test_a_discovery_output_counts_as_produced():
    """Declaring an output that only extract_rows produces must validate."""
    use_case = UseCase(
        name="Discover accounts",
        status="ready",
        allowed_domains=["vendor.test"],
        outputs=["accounts"],
        row_steps=[
            Step(
                id="d1",
                action="extract_rows",
                output="accounts",
                locators=[{"strategy": "css", "selector": "table tbody tr"}],
                columns=[{"name": "account_id", "selector": "td a", "attribute": "href"}],
            )
        ],
    )
    assert use_case.outputs == ["accounts"]


# --- politeness, per use case ---------------------------------------------


def test_the_delay_between_rows_belongs_to_the_use_case():
    """One vendor tolerates a request a second; another refuses after three.

    A single number in the environment cannot be right for both, and the person
    who recorded the workflow is the one who knows which site it is.
    """
    polite = UseCase(name="Slow vendor", row_delay_seconds=5.0)
    assert polite.row_delay_seconds == 5.0
    assert UseCase(name="Unset").row_delay_seconds is None, "None means use the default"

    with pytest.raises(ValueError):
        UseCase(name="Negative", row_delay_seconds=-1)


# --- through the API -------------------------------------------------------


@pytest.fixture
def client(db_settings, db_engine, tmp_path, monkeypatch):
    from conftest import authenticate, build_app

    app = build_app(
        db_settings, tmp_path, monkeypatch,
        session_cls=FakeReplaySession, credentials_key=generate_key(),
    )
    with TestClient(app) as test_client:
        yield authenticate(test_client)


async def seed_discovery(client: TestClient, rows: list[dict[str, str]]) -> str:
    """An execution whose outputs hold what a discovery pass found."""
    from conftest import app_workspace

    store = await app_workspace(client.app)
    usecase_id, _ = await store.save_usecase(
        {
            "id": "disc0000000000000000000000000000",
            "name": "Discover accounts",
            "status": "ready",
            "allowed_domains": ["vendor.test"],
            "outputs": ["accounts"],
            "row_steps": [
                {
                    "id": "d1",
                    "action": "extract_rows",
                    "output": "accounts",
                    "locators": [{"strategy": "css", "selector": "tr"}],
                    "columns": [{"name": "account_id", "selector": "td"}],
                }
            ],
        }
    )
    import uuid

    execution_id = uuid.uuid4().hex
    await store.create_execution(execution_id, usecase_id, 1, inputs={})
    await store.finish_execution(execution_id, status="succeeded", outputs={"accounts": rows})
    return execution_id


async def test_what_discovery_found_becomes_a_dataset(client: TestClient):
    """The join between the two passes.

    Without it the first pass produces a list nobody can act on.
    """
    execution_id = await seed_discovery(
        client,
        [
            {"account_id": "A-1001", "name": "Ada Lovelace"},
            {"account_id": "A-1002", "name": "Grace Hopper"},
        ],
    )

    response = client.post(
        "/api/datasets/from-run",
        json={"execution_id": execution_id, "output": "accounts", "name": "Vendor accounts"},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["row_count"] == 2
    assert [c["name"] for c in body["columns"]] == ["account_id", "name"]
    assert body["sample"][0]["account_id"] == "A-1001"
    assert body["filename"].startswith("run:"), "it records the crawl it came from"


async def test_naming_an_output_that_was_never_extracted_says_what_there_was(
    client: TestClient,
):
    execution_id = await seed_discovery(client, [{"account_id": "A-1"}])

    response = client.post(
        "/api/datasets/from-run",
        json={"execution_id": execution_id, "output": "invoices"},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "'invoices'" in detail
    assert "accounts" in detail, "it lists what the run did produce"


async def test_a_discovery_that_found_nothing_is_refused_rather_than_stored(
    client: TestClient,
):
    """An empty dataset would send a second pass over nothing and look fine."""
    execution_id = await seed_discovery(client, [])

    response = client.post(
        "/api/datasets/from-run",
        json={"execution_id": execution_id, "output": "accounts"},
    )

    assert response.status_code == 422
    assert "no rows" in response.json()["detail"].lower()


async def test_exactly_one_source_must_be_named(client: TestClient):
    both = client.post(
        "/api/datasets/from-run",
        json={"execution_id": "a", "batch_id": "b", "output": "accounts"},
    )
    neither = client.post("/api/datasets/from-run", json={"output": "accounts"})

    assert both.status_code == 422
    assert neither.status_code == 422


# --- downloads -------------------------------------------------------------


def test_a_download_must_name_where_its_file_lands():
    """The file is the point of the step; the name is how a row finds it."""
    with pytest.raises(ValueError) as caught:
        Step(
            id="s1",
            action="download",
            locators=[{"strategy": "css", "selector": "#dl"}],
        )
    assert "output name" in str(caught.value)


def test_a_download_counts_as_producing_its_output():
    use_case = UseCase(
        name="Fetch statements",
        status="ready",
        allowed_domains=["vendor.test"],
        outputs=["statement"],
        row_steps=[
            Step(
                id="s1",
                action="download",
                output="statement",
                locators=[{"strategy": "css", "selector": "#dl"}],
            )
        ],
    )
    assert use_case.outputs == ["statement"]


def test_a_stored_content_type_does_not_depend_on_the_machine():
    """`mimetypes` reads the Windows registry, so .csv differs by platform.

    An artifact kept for years and re-uploaded into another system should not
    carry a content type that depended on which machine fetched it.
    """
    from engine import _mime_for

    assert _mime_for("statement-A-1001.csv") == "text/csv"
    assert _mime_for("deed.PDF") == "application/pdf"
    assert _mime_for("ledger.xlsx") == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert _mime_for("mystery.qqq") == "application/octet-stream"


def test_a_downloaded_name_cannot_forge_a_response_header():
    """The filename comes off a vendor's site, so it is untrusted input."""
    from routers.runs import _artifact_headers

    class Record:
        filename = 'evil".pdf\r\nX-Injected: yes'

    headers = _artifact_headers(Record())
    disposition = headers["Content-Disposition"]
    assert "\r" not in disposition and "\n" not in disposition
    assert "X-Injected" not in disposition.split(";")[0]


def test_a_screenshot_is_not_served_as_an_attachment():
    """It is rendered inline in the trail; only a download has a name."""
    from routers.runs import _artifact_headers

    class Record:
        filename = ""

    assert "Content-Disposition" not in _artifact_headers(Record())


# --- pointing at a field while recording -----------------------------------

POINTED = '''import asyncio
from playwright.async_api import Playwright, async_playwright, expect


async def run(playwright: Playwright) -> None:
    browser = await playwright.chromium.launch(headless=False)
    context = await browser.new_context()
    page = await context.new_page()
    await page.goto("https://vendor.test/account")
    await expect(page.get_by_label("Balance")).to_have_text("1,240.55")
    await page.get_by_role("link", name="Statements").click()
    await expect(page.get_by_label("Reference")).to_have_value("REF-88213")
'''


def test_pointing_at_a_field_is_captured_with_its_locator():
    """The only point-and-click gesture codegen offers.

    "Assert text" and "Assert value" are how a non-technical person names an
    element without typing a selector. The text case used to keep the value and
    throw the locator away -- leaving "this text is somewhere on the page",
    which cannot be read from -- and the value case was not parsed at all.
    """
    from codegen import parse as parse_script

    recording = parse_script(POINTED)

    assert [(c.kind, c.locator.text, c.value) for c in recording.captured] == [
        ("text", "Balance", "1,240.55"),
        ("value", "Reference", "REF-88213"),
    ]
    assert not recording.unsupported, recording.unsupported


def test_a_named_field_is_read_where_it_was_pointed_at():
    """Position is correctness, not tidiness.

    Appending every reading to the end would read the first page's field after
    the browser had already moved to the third.
    """
    from codegen import parse as parse_script
    from fields import FieldSet
    from routers.recordings import build_usecase

    recording = parse_script(POINTED)
    lines = [c.line for c in recording.captured]
    use_case = build_usecase(
        recording,
        name="Read an account",
        description="",
        declared=FieldSet.from_payload([]),
        extractions={lines[0]: "balance", lines[1]: "reference"},
    )

    assert [(s.action, s.output) for s in use_case.row_steps if s.action != "assert"] == [
        ("navigate", None),
        ("extract", "balance"),
        ("click", None),
        ("extract", "reference"),
    ], "each value is read on the page it was pointed at"
    assert use_case.outputs == ["balance", "reference"]


def test_an_input_is_read_by_its_value_not_its_text():
    """`inner_text` on an input returns nothing, so the button used decides."""
    from codegen import parse as parse_script
    from fields import FieldSet
    from routers.recordings import build_usecase

    recording = parse_script(POINTED)
    lines = [c.line for c in recording.captured]
    use_case = build_usecase(
        recording, name="x", description="", declared=FieldSet.from_payload([]),
        extractions={lines[0]: "balance", lines[1]: "reference"},
    )

    by_output = {s.output: s for s in use_case.row_steps if s.action == "extract"}
    assert by_output["balance"].attribute == "", "visible text is read as text"
    assert by_output["reference"].attribute == "value", "a field holds its text in value"


def test_leaving_a_pointed_at_element_unnamed_keeps_it_as_a_check():
    """Not everything pointed at is a value; some of it is "this should say X"."""
    from codegen import parse as parse_script
    from fields import FieldSet
    from routers.recordings import build_usecase

    recording = parse_script(POINTED)
    use_case = build_usecase(
        recording, name="x", description="", declared=FieldSet.from_payload([]),
        extractions={},
    )

    assert not [s for s in use_case.all_steps if s.action == "extract"]
    assert use_case.outputs == []
    assert [s for s in use_case.all_steps if s.action == "assert"], "still checked"
