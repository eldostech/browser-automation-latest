"""Domain allowlist and sensitive-action classification."""

from __future__ import annotations

import pytest

from policy import Category, check_navigation, classify, domain_allowed, extract_urls

ALLOWLIST = ["example.com", "*.shop.example.org"]


# --- allowlist -------------------------------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://example.com", True),
        ("https://example.com/deep/path?q=1", True),
        ("http://example.com", True),
        ("https://EXAMPLE.COM/Path", True),
        ("https://example.com:8443/x", True),
        # Exact entries do not cover subdomains.
        ("https://www.example.com", False),
        # Wildcards cover subdomains and the apex.
        ("https://shop.example.org", True),
        ("https://eu.shop.example.org", True),
        # Look-alikes must not slip through.
        ("https://notexample.com", False),
        ("https://example.com.evil.net", False),
        ("https://evil.net/?next=https://example.com", False),
        # Non-http schemes are refused outright.
        ("file:///etc/passwd", False),
        ("data:text/html,<h1>hi</h1>", False),
        ("about:blank", False),
    ],
)
def test_domain_allowed(url: str, expected: bool):
    assert domain_allowed(url, ALLOWLIST) is expected


def test_empty_allowlist_denies_everything():
    assert domain_allowed("https://example.com", []) is False


def test_star_disables_the_allowlist():
    assert domain_allowed("https://anything.example.net", ["*"]) is True


def test_check_navigation_flags_the_offending_url():
    result = check_navigation("browser_navigate", {"url": "https://evil.net"}, ALLOWLIST)
    assert result.allowed is False
    assert result.url == "https://evil.net"
    assert "evil.net" in result.reason


def test_check_navigation_passes_tools_without_urls():
    assert check_navigation("browser_snapshot", {}, ALLOWLIST).allowed is True


def test_urls_are_found_in_nested_arguments():
    args = {"options": {"items": [{"href": "https://evil.net/page"}]}}
    assert "https://evil.net/page" in extract_urls(args)
    assert check_navigation("browser_click", args, ALLOWLIST).allowed is False


@pytest.mark.parametrize(
    "value",
    [
        "ref=e15",
        "[ref=e15]",
        'button[name="play"]',
        "_blank",
        "Open navigation menu button",
    ],
)
def test_element_refs_in_url_keys_are_not_navigation(value: str):
    """A selector under a URL-named key is not a navigation attempt."""
    assert extract_urls({"target": value}) == []
    assert check_navigation("browser_click", {"target": value}, ALLOWLIST).allowed is True


@pytest.mark.parametrize(
    "value",
    ["https://evil.net[", "http://[::1", "//evil.net/page", "javascript:alert(1)"],
)
def test_malformed_urls_are_denied_not_crashed(value: str):
    """urlparse raises on some malformed input; the policy must still answer."""
    assert domain_allowed(value, ALLOWLIST) is False


# --- classification --------------------------------------------------------


def test_ordinary_navigation_is_not_sensitive():
    decision = classify("browser_navigate", {"url": "https://example.com"}, allowlist=ALLOWLIST)
    assert decision.sensitive is False
    assert decision.categories == []


def test_snapshot_is_not_sensitive():
    assert classify("browser_snapshot", {}, allowlist=ALLOWLIST).sensitive is False


def test_password_entry_is_sensitive():
    decision = classify(
        "browser_type",
        {"element": "Password field", "ref": "e4", "text": "hunter2"},
        allowlist=ALLOWLIST,
    )
    assert decision.sensitive is True
    assert Category.CREDENTIALS in decision.categories


def test_checkout_click_is_sensitive():
    decision = classify(
        "browser_click", {"element": "Place order button", "ref": "e9"}, allowlist=ALLOWLIST
    )
    assert decision.sensitive is True
    assert Category.PAYMENT in decision.categories


def test_delete_click_is_sensitive():
    decision = classify(
        "browser_click", {"element": "Delete account", "ref": "e2"}, allowlist=ALLOWLIST
    )
    assert Category.DESTRUCTIVE in decision.categories


def test_form_submit_click_is_sensitive():
    decision = classify("browser_click", {"element": "Submit form"}, allowlist=ALLOWLIST)
    assert Category.FORM_SUBMIT in decision.categories


def test_enter_key_counts_as_a_form_submit():
    decision = classify("browser_press_key", {"key": "Enter"}, allowlist=ALLOWLIST)
    assert Category.FORM_SUBMIT in decision.categories


def test_arrow_key_does_not():
    assert classify("browser_press_key", {"key": "ArrowDown"}, allowlist=ALLOWLIST).sensitive is False


def test_javascript_evaluation_is_always_gated():
    decision = classify("browser_evaluate", {"function": "() => 1 + 1"}, allowlist=ALLOWLIST)
    assert Category.CODE_EXECUTION in decision.categories


def test_file_upload_is_always_gated():
    decision = classify("browser_file_upload", {"paths": ["/tmp/a.pdf"]}, allowlist=ALLOWLIST)
    assert Category.FILE_UPLOAD in decision.categories


def test_off_allowlist_navigation_is_sensitive():
    decision = classify("browser_navigate", {"url": "https://evil.net"}, allowlist=ALLOWLIST)
    assert Category.OFF_ALLOWLIST in decision.categories
    assert decision.sensitive is True


def test_reason_is_human_readable():
    decision = classify(
        "browser_click", {"element": "Pay now with saved card"}, allowlist=ALLOWLIST
    )
    assert decision.reason
    assert "payment" in decision.reason.lower()


def test_categories_are_deduplicated():
    decision = classify(
        "browser_click",
        {"element": "Delete and remove account", "text": "delete"},
        allowlist=ALLOWLIST,
    )
    assert len(decision.categories) == len(set(decision.categories))


# --- glob patterns in the allowlist ----------------------------------------


@pytest.mark.parametrize(
    "url,allowed",
    [
        ("http://localhost:8000", True),
        ("http://localhost:8000/login", True),
        ("http://my-localhost-box:3000", True),
        ("https://example.com", False),
    ],
)
def test_a_wildcard_pattern_is_matched_as_a_glob(url: str, allowed: bool):
    """`*localhost*` used to match nothing at all.

    Only two forms were understood -- an exact host and a leading `*.` -- so a
    pattern with wildcards anywhere else was silently inert, while the refusal
    message listed it back verbatim. The allowlist appeared to contradict
    itself: "host is not in the allowed domain list (*localhost*)".
    """
    assert domain_allowed(url, ["*localhost*"]) is allowed


def test_a_glob_matches_substrings_which_is_what_it_asks_for():
    """Worth knowing rather than surprising: `*localhost*` is broad."""
    assert domain_allowed("https://notlocalhost.evil.com", ["*localhost*"]) is True
    # An exact host is the narrower thing most people mean.
    assert domain_allowed("https://notlocalhost.evil.com", ["localhost"]) is False


def test_subdomain_patterns_still_match_the_apex():
    """The `*.` form keeps its special meaning rather than becoming a glob."""
    assert domain_allowed("https://example.com", ["*.example.com"]) is True
    assert domain_allowed("https://a.b.example.com", ["*.example.com"]) is True


def test_globs_do_not_weaken_the_scheme_or_empty_list_rules():
    assert domain_allowed("about:blank", ["*localhost*"]) is False
    assert domain_allowed("file:///etc/passwd", ["*"]) is False
    assert domain_allowed("http://localhost:8000", []) is False
