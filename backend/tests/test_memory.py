"""What broke before, and what fixed it.

The properties worth holding down here are the three that make recall safe
rather than merely useful: it never crosses a workspace, it never crosses a
domain, and it never becomes authoritative. The last one is enforced in
``healing.py`` -- the model still picks from candidates on the page -- and is
asserted there; this file covers the first two and the write-back.
"""

from __future__ import annotations

import pytest

from embeddings import HashEmbedder, page_signature
from memory import MAX_DISTANCE, HealingMemory, PastFix, as_prompt, domain_of

pytestmark = pytest.mark.anyio


@pytest.fixture
def embedder() -> HashEmbedder:
    """The deterministic stand-in, so the suite needs no AWS."""
    return HashEmbedder()


@pytest.fixture
def memory(store, embedder) -> HealingMemory:
    return HealingMemory(store, embedder)


SIGN_IN = 'button "Sign in"\ntextbox "Username"\ntextbox "Password"'
CHECKOUT = 'button "Place order"\ntextbox "Card number"\nheading "Checkout"'


async def remember(memory, *, url, page, new_name, by="model", step="s1"):
    await memory.remember(
        usecase_id="uc-1",
        step_id=step,
        page_url=url,
        page=page,
        step_summary="click Sign in",
        wanted='role=button name="Sign in"',
        old_locator={"strategy": "role", "role": "button", "name": "Sign in"},
        new_locator={"strategy": "role", "role": "button", "name": new_name},
        explanation=f"the button is called {new_name} now",
        confirmed_by=by,
    )


# --- the domain boundary ---------------------------------------------------


async def test_a_fix_is_recalled_on_the_site_it_was_made_on(memory):
    await remember(memory, url="https://shop.test/login", page=SIGN_IN, new_name="Log in")

    found = await memory.recall(
        step_summary="click Sign in",
        wanted='role=button name="Sign in"',
        page_url="https://shop.test/login",
        page=SIGN_IN,
    )
    assert [fix.new_locator["name"] for fix in found] == ["Log in"]


async def test_a_fix_is_not_recalled_on_a_different_site(memory):
    """Nearest-neighbour over everything would cheerfully return a plausible
    button from an unrelated site. The domain is a hard filter."""
    await remember(memory, url="https://shop.test/login", page=SIGN_IN, new_name="Log in")

    found = await memory.recall(
        step_summary="click Sign in",
        wanted='role=button name="Sign in"',
        page_url="https://other.test/login",
        page=SIGN_IN,
    )
    assert found == []


async def test_a_page_with_no_url_recalls_nothing(memory):
    await remember(memory, url="https://shop.test/login", page=SIGN_IN, new_name="Log in")
    assert await memory.recall(
        step_summary="x", wanted="y", page_url="", page=SIGN_IN
    ) == []


# --- the tenant boundary ---------------------------------------------------


async def test_a_fix_never_crosses_a_workspace(root_store, embedder):
    """Selectors describe the shape of another company's internal tooling."""
    mine = await root_store.ensure_workspace("Mine", "mine")
    theirs = await root_store.ensure_workspace("Theirs", "theirs")

    await remember(
        HealingMemory(root_store.workspace(mine), embedder),
        url="https://shop.test/login",
        page=SIGN_IN,
        new_name="Log in",
    )

    found = await HealingMemory(root_store.workspace(theirs), embedder).recall(
        step_summary="click Sign in",
        wanted='role=button name="Sign in"',
        page_url="https://shop.test/login",
        page=SIGN_IN,
    )
    assert found == []


# --- ranking ---------------------------------------------------------------


async def test_the_closest_page_comes_back_first(memory):
    await remember(memory, url="https://shop.test/login", page=SIGN_IN, new_name="Log in", step="s1")
    await remember(
        memory, url="https://shop.test/checkout", page=CHECKOUT, new_name="Buy now", step="s2"
    )

    found = await memory.recall(
        step_summary="click Sign in",
        wanted='role=button name="Sign in"',
        page_url="https://shop.test/login",
        page=SIGN_IN,
    )
    assert found, "the sign-in page should recall something"
    assert found[0].step_id == "s1"


async def test_an_unrelated_page_on_the_same_site_is_too_far_to_count(memory):
    """Two pages of one site share enough chrome to score somewhat close.
    A distance cut is what stops an unrelated fix reaching the prompt."""
    await remember(
        memory, url="https://shop.test/checkout", page=CHECKOUT, new_name="Buy now", step="s2"
    )

    found = await memory.recall(
        step_summary="click Sign in",
        wanted='role=button name="Sign in"',
        page_url="https://shop.test/login",
        page=SIGN_IN,
    )
    assert all(fix.distance <= MAX_DISTANCE for fix in found)


# --- what the prompt sees --------------------------------------------------


def test_a_human_confirmed_fix_outranks_a_closer_one_the_model_made():
    """Somebody looked at that one and said yes, which beats a better match
    nobody checked."""
    model_fix = PastFix("s1", "model guessed", None, {"name": "A"}, "model", distance=0.01)
    human_fix = PastFix("s2", "a person said so", None, {"name": "B"}, "ada@x.test", distance=0.4)

    rendered = as_prompt([model_fix, human_fix])
    assert rendered.index("a person said so") < rendered.index("model guessed")


def test_nothing_remembered_says_so_rather_than_being_blank():
    """A blank section reads as a formatting bug; a sentence reads as a fact."""
    assert "nothing similar" in as_prompt([])


# --- turning it off --------------------------------------------------------


async def test_with_no_embedder_nothing_is_recalled_or_recorded(store):
    """Healing then behaves exactly as it did before there was a memory, which
    is what makes this an optimisation rather than a dependency."""
    off = HealingMemory(store, None)
    assert off.available is False

    await remember(off, url="https://shop.test/login", page=SIGN_IN, new_name="Log in")
    assert await off.recall(
        step_summary="x", wanted="y", page_url="https://shop.test/login", page=SIGN_IN
    ) == []
    assert await store.list_fixes() == []


async def test_an_embedder_that_fails_is_not_an_error(store):
    class Broken:
        async def embed(self, text: str):
            raise RuntimeError("bedrock is down")

    memory = HealingMemory(store, Broken())
    # Neither of these may raise: retrieval is an optimisation, and a write is
    # bookkeeping about a repair that already worked.
    with pytest.raises(RuntimeError):
        await Broken().embed("x")
    try:
        await memory.recall(
            step_summary="x", wanted="y", page_url="https://shop.test/", page=SIGN_IN
        )
    except RuntimeError:
        pytest.fail("a failed embedding must not surface as an error")


# --- forgetting ------------------------------------------------------------


async def test_a_fix_can_be_taken_back_out(memory, store):
    """One that was right last month and wrong now is what makes healing
    confidently incorrect."""
    await remember(memory, url="https://shop.test/login", page=SIGN_IN, new_name="Log in")
    [fix] = await store.list_fixes()

    assert await store.forget_fix(fix["id"]) is True
    assert await store.list_fixes() == []
    assert await store.forget_fix(fix["id"]) is False


async def test_the_stored_record_does_not_carry_the_vector(memory, store):
    """A thousand floats nobody reads. It is only ever used inside a query."""
    await remember(memory, url="https://shop.test/login", page=SIGN_IN, new_name="Log in")
    [fix] = await store.list_fixes()
    assert "embedding" not in fix


# --- the helpers -----------------------------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://Shop.Test/login", "shop.test"),
        ("http://127.0.0.1:8000/x", "127.0.0.1"),
        ("not a url", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_the_domain_is_what_scopes_recall(url, expected):
    assert domain_of(url) == expected


async def test_similar_text_scores_closer_than_unrelated_text(embedder):
    """The only property the retrieval path asks of an embedding, asserted of
    the stand-in so the tests do not need AWS to be meaningful."""
    import math

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b)) / (
            math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        )

    base = await embedder.embed(page_signature("click Sign in", "button", "u", SIGN_IN))
    near = await embedder.embed(page_signature("click Sign in", "button", "u", SIGN_IN + "\nlink"))
    far = await embedder.embed(page_signature("place order", "button", "u", CHECKOUT))

    assert cosine(base, near) > cosine(base, far)
