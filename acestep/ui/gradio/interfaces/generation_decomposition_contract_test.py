"""AST contract tests for generation interface decomposition."""

import ast
from pathlib import Path
import unittest


_INTERFACES_DIR = Path(__file__).resolve().parent
_WIRING_DIR = _INTERFACES_DIR.parent / "events" / "wiring"


def _load_module(module_name: str) -> ast.Module:
    """Parse and return AST for a generation interface module.

    Args:
        module_name: Filename under ``acestep/ui/gradio/interfaces`` to parse.

    Returns:
        Parsed ``ast.Module`` tree for the requested module.
    """

    path = _INTERFACES_DIR / module_name
    return ast.parse(path.read_text(encoding="utf-8"))


def _call_name(node: ast.AST) -> str | None:
    """Extract a simple call-target name from an AST call function node.

    Args:
        node: AST node representing a call target expression.

    Returns:
        The simple function/attribute name when resolvable; otherwise ``None``.
    """

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _collect_return_dict_keys(module_name: str, function_name: str) -> set[str]:
    """Collect string keys from dict literals assigned/returned in a function.

    Args:
        module_name: Module filename under ``interfaces`` to inspect.
        function_name: Function name whose body should be scanned.

    Returns:
        Set of string keys found in literal dict assignments/updates/returns.
    """

    module = _load_module(module_name)
    function_node = None
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            function_node = node
            break
    if function_node is None:
        raise AssertionError(f"{function_name} not found in {module_name}")

    keys: set[str] = set()
    for node in ast.walk(function_node):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Dict):
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "update":
            if node.args and isinstance(node.args[0], ast.Dict):
                for key in node.args[0].keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        keys.add(key.value)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
    return keys


def _collect_generation_section_keys_used_by_wiring() -> set[str]:
    """Collect generation-section keys referenced by wiring modules.

    Args:
        None.

    Returns:
        Set of ``generation_section[...]`` keys consumed by wiring modules.
    """

    keys: set[str] = set()
    for path in _WIRING_DIR.glob("*.py"):
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(module):
            if not isinstance(node, ast.Subscript):
                continue
            if not isinstance(node.value, ast.Name) or node.value.id != "generation_section":
                continue
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                keys.add(node.slice.value)
    return keys


class GenerationDecompositionContractTests(unittest.TestCase):
    """Verify that generation interface facade delegates to focused helpers."""

    def test_generation_facade_imports_and_wraps_new_helpers(self):
        """`generation.py` should import helper modules and merge section dicts."""

        module = _load_module("generation.py")
        imported_modules = []
        call_names = []
        update_calls = 0

        for node in ast.walk(module):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name:
                    call_names.append(name)
                if isinstance(node.func, ast.Attribute) and node.func.attr == "update":
                    update_calls += 1

        self.assertIn("generation_advanced_settings", imported_modules)
        self.assertIn("generation_service_config", imported_modules)
        self.assertIn("generation_tab_section", imported_modules)
        self.assertIn("create_advanced_settings_section", call_names)
        self.assertIn("create_generation_tab_section", call_names)
        self.assertGreaterEqual(update_calls, 2)

    def test_generation_facade_exposes_required_public_symbols(self):
        """Facade should expose key constructors imported by interfaces package."""

        module = _load_module("generation.py")
        names = set()
        for node in module.body:
            if isinstance(node, ast.FunctionDef):
                names.add(node.name)
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    names.add(alias.asname or alias.name)

        self.assertIn("create_advanced_settings_section", names)
        self.assertIn("create_generation_tab_section", names)
        self.assertIn("create_service_config_section", names)

    def test_advanced_settings_section_delegates_to_control_builders(self):
        """Advanced settings should compose service/lora/dit/lm/output helpers."""

        module = _load_module("generation_advanced_settings.py")
        call_names = []
        for node in ast.walk(module):
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name:
                    call_names.append(name)

        self.assertIn("create_service_config_content", call_names)
        self.assertIn("build_lora_controls", call_names)
        self.assertIn("build_dit_controls", call_names)
        self.assertIn("build_lm_controls", call_names)
        self.assertIn("build_output_controls", call_names)
        self.assertIn("build_automation_controls", call_names)

    def test_generation_tab_section_delegates_to_component_builders(self):
        """Generation tab should compose focused primary/secondary/runtime helpers."""

        module = _load_module("generation_tab_section.py")
        call_names = []
        for node in ast.walk(module):
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name:
                    call_names.append(name)

        self.assertIn("build_mode_selector_controls", call_names)
        self.assertIn("build_hidden_generation_state", call_names)
        self.assertIn("build_simple_mode_controls", call_names)
        self.assertIn("build_source_track_and_code_controls", call_names)
        self.assertIn("build_cover_strength_controls", call_names)
        self.assertIn("build_custom_mode_controls", call_names)
        self.assertIn("build_repainting_controls", call_names)
        self.assertIn("build_optional_parameter_controls", call_names)
        self.assertIn("build_generate_row_controls", call_names)

    def test_service_config_section_delegates_to_row_and_toggle_helpers(self):
        """Service config section should compose row/toggle helper builders."""

        module = _load_module("generation_service_config.py")
        call_names = []
        for node in ast.walk(module):
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name:
                    call_names.append(name)

        self.assertIn("build_language_selector", call_names)
        self.assertIn("build_gpu_info_and_tier", call_names)
        self.assertIn("build_checkpoint_controls", call_names)
        self.assertIn("build_model_device_controls", call_names)
        self.assertIn("build_lm_backend_controls", call_names)
        self.assertIn("build_service_toggles", call_names)
        self.assertIn("build_service_init_controls", call_names)

    def test_generation_keys_cover_all_wiring_generation_section_requirements(self):
        """Returned generation keys should cover all keys consumed by wiring modules."""

        produced_keys: set[str] = set()
        key_sources = [
            ("generation_tab_section.py", "create_generation_tab_section"),
            ("generation_advanced_settings.py", "create_advanced_settings_section"),
            ("generation_service_config.py", "create_service_config_content"),
            ("generation_tab_primary_controls.py", "build_mode_selector_controls"),
            ("generation_tab_primary_controls.py", "build_hidden_generation_state"),
            ("generation_tab_primary_controls.py", "build_simple_mode_controls"),
            ("generation_tab_primary_controls.py", "build_source_track_and_code_controls"),
            ("generation_tab_secondary_controls.py", "build_cover_strength_controls"),
            ("generation_tab_secondary_controls.py", "build_custom_mode_controls"),
            ("generation_tab_secondary_controls.py", "build_repainting_controls"),
            ("generation_tab_runtime_controls.py", "build_optional_parameter_controls"),
            ("generation_tab_runtime_controls.py", "build_generate_row_controls"),
            ("generation_advanced_dit_controls.py", "build_dit_controls"),
            ("generation_advanced_primary_controls.py", "build_lm_controls"),
            ("generation_advanced_primary_controls.py", "build_lora_controls"),
            ("generation_advanced_output_controls.py", "build_output_controls"),
            ("generation_advanced_output_controls.py", "build_automation_controls"),
            ("generation_service_config_rows.py", "build_language_selector"),
            ("generation_service_config_rows.py", "build_gpu_info_and_tier"),
            ("generation_service_config_rows.py", "build_checkpoint_controls"),
            ("generation_service_config_rows.py", "build_model_device_controls"),
            ("generation_service_config_rows.py", "build_lm_backend_controls"),
            ("generation_service_config_toggles.py", "build_service_toggles"),
            ("generation_service_config_toggles.py", "build_service_init_controls"),
        ]
        for module_name, function_name in key_sources:
            produced_keys |= _collect_return_dict_keys(module_name, function_name)
        produced_keys.discard("device_value")

        required_keys = _collect_generation_section_keys_used_by_wiring()
        self.assertTrue(
            required_keys.issubset(produced_keys),
            f"Missing generation_section keys: {sorted(required_keys - produced_keys)}",
        )


if __name__ == "__main__":
    unittest.main()
