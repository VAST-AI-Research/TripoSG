import unittest

from aethryx_tools import (
    analyze_project_diagnostics,
    execute_tool,
    get_project_context,
)


class AethryxToolTests(unittest.TestCase):
    def test_project_context_requires_identity(self):
        result = get_project_context({})
        self.assertEqual(result["status"], "unavailable")

    def test_diagnostics_use_real_project_state(self):
        context = {
            "project_id": "project-1",
            "revision": "4",
            "references": [],
            "mesh_stats": {"vertices": 2506, "faces": 5000},
            "model": "mesh.glb",
        }
        result = analyze_project_diagnostics(context)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["revision"], "4")
        self.assertTrue(any("2506 vertices" in finding for finding in result["findings"]))

    def test_unregistered_tool_is_not_executed(self):
        result = execute_tool("shell", {}, {})
        self.assertEqual(result["status"], "unavailable")

    def test_write_permission_is_not_granted_by_default(self):
        result = execute_tool("validate_avatar", {}, {}, write_allowed=False)
        self.assertEqual(result["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
