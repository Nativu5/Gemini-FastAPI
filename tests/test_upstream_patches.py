"""Corrections that `app.services.client` applies to gemini-webapi when it is imported."""

from gemini_webapi import GeminiClient

import app.services.client  # noqa: F401


def test_the_link_google_leaves_for_a_generated_image_is_removed_whole():
    """Parsed by the library itself, so this also fails if it stops reading the pattern that
    is replaced."""
    raw = "Here it is:\n\nhttp://googleusercontent.com/image_generation_content/0_974\n\nEnjoy."

    text = GeminiClient("psid", "psidts")._parse_candidate([None, [raw]], "c", "r", "rc")[0]

    assert text == "Here it is:\n\nEnjoy."
