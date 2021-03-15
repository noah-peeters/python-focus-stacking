from QtUi import utilities as Utilities
import pytest

def test_format_seconds():
    assert Utilities().format_seconds(65432) == '18:10:32'