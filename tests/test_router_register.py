"""
Tests for the public ``register`` function in simplemem_router.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import simplemem_router as simplemem
from simplemem_router import register, list_modes, is_available


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_MODE = "_test_register_dummy"
_TEST_MODE_2 = "_test_register_dummy2"


def _cleanup(*modes):
    """Remove test modes from the registry after each test."""
    # Access the private registry to clean up test entries.
    import simplemem_router as _sr
    for mode in modes:
        _sr._registry.pop(mode, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_adds_mode_to_list_modes(self):
        """A newly registered mode appears in list_modes()."""
        try:
            register(
                mode=_TEST_MODE,
                module_path="os",
                class_name="getcwd",
                description="Dummy test backend",
            )
            assert _TEST_MODE in list_modes()
            assert list_modes()[_TEST_MODE] == "Dummy test backend"
        finally:
            _cleanup(_TEST_MODE)

    def test_register_duplicate_raises_value_error(self):
        """Registering an existing mode without overwrite=True raises ValueError."""
        try:
            register(
                mode=_TEST_MODE,
                module_path="os",
                class_name="getcwd",
                description="First",
            )
            with pytest.raises(ValueError, match="already registered"):
                register(
                    mode=_TEST_MODE,
                    module_path="os",
                    class_name="getcwd",
                    description="Second",
                )
        finally:
            _cleanup(_TEST_MODE)

    def test_register_overwrite_replaces_description(self):
        """overwrite=True silently replaces the existing registration."""
        try:
            register(
                mode=_TEST_MODE,
                module_path="os",
                class_name="getcwd",
                description="Original",
            )
            register(
                mode=_TEST_MODE,
                module_path="os",
                class_name="getcwd",
                description="Replaced",
                overwrite=True,
            )
            assert list_modes()[_TEST_MODE] == "Replaced"
        finally:
            _cleanup(_TEST_MODE)

    def test_register_stores_required_deps(self):
        """required_deps are stored on the backend descriptor."""
        import simplemem_router as _sr
        try:
            register(
                mode=_TEST_MODE,
                module_path="os",
                class_name="getcwd",
                required_deps=["os"],
            )
            assert _sr._registry[_TEST_MODE].required_deps == ["os"]
        finally:
            _cleanup(_TEST_MODE)

    def test_register_missing_dep_not_available(self):
        """is_available returns False when a required package is absent."""
        try:
            register(
                mode=_TEST_MODE,
                module_path="os",
                class_name="getcwd",
                required_deps=["_nonexistent_package_xyz"],
            )
            assert is_available(_TEST_MODE) is False
        finally:
            _cleanup(_TEST_MODE)
