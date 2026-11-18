"""Fetching a URL a client put in a request: what may be fetched, and how much of it."""

import asyncio
import socket

import pytest

from app.utils import g_config, helper
from app.utils.helper import reject_unsafe_url, save_url_to_tempfile

PUBLIC, PRIVATE = "93.184.216.34", "10.0.0.7"


@pytest.fixture(autouse=True)
def _names_resolve_without_dns(monkeypatch):
    """`*.private.test` resolves into the local network, any other name to a public address."""

    def getaddrinfo(host: str, *args, **kwargs):
        try:
            socket.inet_pton(socket.AF_INET6 if ":" in host else socket.AF_INET, host)
            address = host
        except OSError:
            address = PRIVATE if host.endswith(".private.test") else PUBLIC
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (address, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(g_config.gemini, "allow_private_url_fetch", False)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x",
        "http://[::1]/x",
        "http://10.0.0.5/x",
        "http://192.168.1.1/x",
        "http://169.254.169.254/latest/meta-data",  # cloud metadata
        "http://0.0.0.0/x",
        "http://224.0.0.1/x",  # multicast, which `ipaddress` calls global
        "http://intranet.private.test/x",  # a public-looking name that resolves inward
    ],
)
def test_a_url_that_leads_into_the_local_network_is_refused(url):
    with pytest.raises(ValueError, match="Refusing to fetch private/reserved address"):
        reject_unsafe_url(url)


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("file:///etc/passwd", "Unsupported URL scheme"),
        ("ftp://example.com/x", "Unsupported URL scheme"),
        ("http:///x", "must include a hostname"),
    ],
)
def test_only_http_urls_with_a_host_are_fetched(url, reason):
    with pytest.raises(ValueError, match=reason):
        reject_unsafe_url(url)


def test_a_public_url_is_allowed_and_the_local_network_only_on_request(monkeypatch):
    reject_unsafe_url("https://example.com/cat.png")

    monkeypatch.setattr(g_config.gemini, "allow_private_url_fetch", True)
    reject_unsafe_url("http://127.0.0.1:8000/cat.png")
    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        reject_unsafe_url("file:///etc/passwd")  # the switch is about addresses, not schemes


class _Download:
    """Stands in for the HTTP session: serves `body` in 4-byte chunks and counts its uses."""

    body = b""
    opened = 0

    def __init__(self, **kwargs):
        type(self).opened += 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url: str, content_callback):
        for start in range(0, len(self.body), 4):
            content_callback(self.body[start : start + 4])
        return type(
            "Response",
            (),
            {"headers": {"content-type": "image/png"}, "raise_for_status": lambda self: None},
        )()


@pytest.fixture
def download(monkeypatch):
    monkeypatch.setattr(helper.requests, "AsyncSession", _Download)
    monkeypatch.setattr(_Download, "opened", 0)
    return _Download


def test_a_refused_url_is_never_requested(download, tmp_path):
    with pytest.raises(ValueError, match="Refusing to fetch"):
        asyncio.run(save_url_to_tempfile("http://intranet.private.test/secret", tmp_path))

    assert download.opened == 0
    assert list(tmp_path.iterdir()) == []


def test_a_download_is_saved_under_the_extension_of_its_content_type(
    download, tmp_path, monkeypatch
):
    monkeypatch.setattr(download, "body", b"12345678")

    saved = asyncio.run(save_url_to_tempfile("https://example.com/picture", tmp_path))

    assert (saved.read_bytes(), saved.suffix) == (b"12345678", ".png")


def test_a_download_larger_than_the_cap_is_abandoned_and_leaves_no_file(
    download, tmp_path, monkeypatch
):
    monkeypatch.setattr(helper, "MAX_REMOTE_FETCH_BYTES", 10)
    monkeypatch.setattr(download, "body", b"x" * 11)

    with pytest.raises(ValueError, match="exceeded 10 bytes"):
        asyncio.run(save_url_to_tempfile("https://example.com/huge.bin", tmp_path))

    assert list(tmp_path.iterdir()) == []


def test_a_data_url_is_decoded_locally(download, tmp_path):
    saved = asyncio.run(save_url_to_tempfile("data:image/png;base64,aGVsbG8=", tmp_path))

    assert (saved.read_bytes(), saved.suffix) == (b"hello", ".png")
    assert download.opened == 0
