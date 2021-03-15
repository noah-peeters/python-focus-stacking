from focus_stack import utilities
import pytest


def test_format_seconds():
    assert utilities().format_seconds(65432) == "18:10:32"
