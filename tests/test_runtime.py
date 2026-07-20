import tempfile
import unittest
from pathlib import Path

from agent_runtime import (
    AgentRuntime,
    ContextManager,
    DemoClient,
    Session,
    SessionStore,
    ToolCall,
    make_registry,
    safe_calculator,
)


class RuntimeTests(unittest.TestCase):
    def make_runtime(self, root: Path) -> AgentRuntime:
        return AgentRuntime(
            model=DemoClient(),
            registry=make_registry(root / "knowledge"),
            sessions=SessionStore(root / "sessions"),
        )

    def test_calculator_rejects_names(self):
        self.assertEqual(safe_calculator("12 * (7 + 1)"), "96")
        with self.assertRaises(ValueError):
            safe_calculator("__import__('os').getcwd()")

    def test_demo_tool_loop_and_persistence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "knowledge").mkdir()
            (root / "knowledge" / "s20.md").write_text("S20 agent loop", encoding="utf-8")
            runtime = self.make_runtime(root)
            result = runtime.run("demo", "计算 12 * 7")
            self.assertIn("84", result["answer"])
            self.assertTrue(any(item["kind"] == "tool_call" for item in result["trace"]))
            saved = runtime.sessions.load("demo")
            self.assertEqual(saved.messages[-1]["role"], "assistant")
            second = runtime.run("demo", "search S20")
            self.assertIn("s20.md", second["answer"])
            self.assertTrue(any(item["kind"] == "tool_call" for item in second["trace"]))

    def test_search_tool_is_rooted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "knowledge").mkdir()
            (root / "knowledge" / "s20.md").write_text("agent loop", encoding="utf-8")
            registry = make_registry(root / "knowledge")
            output = registry.execute(
                ToolCall("search", "search", {"query": "agent"})
            )
            self.assertIn("s20.md", output)
            blocked = registry.execute(
                ToolCall("read", "read_docs", {"path": "../secret"})
            )
            self.assertIn("escapes docs root", blocked)

    def test_context_keeps_tool_pair(self):
        session = Session(
            id="x",
            messages=[
                {"role": "user", "content": "old"},
                {"role": "assistant", "tool_calls": [{"id": "1"}]},
                {"role": "tool", "tool_call_id": "1", "content": "result"},
                {"role": "user", "content": "new"},
            ],
        )
        ContextManager(recent_messages=2).compact(session)
        self.assertNotEqual(session.messages[0]["role"], "tool")

    def test_session_create_and_list(self):
        with tempfile.TemporaryDirectory() as folder:
            store = SessionStore(Path(folder))
            created = store.create("interview_session")
            created.messages.append({"role": "user", "content": "准备 Agent 面试"})
            store.save(created)
            sessions = store.list_sessions()
            self.assertEqual(sessions[0]["id"], "interview_session")
            self.assertEqual(sessions[0]["preview"], "准备 Agent 面试")
            with self.assertRaises(ValueError):
                store.create("interview_session")


if __name__ == "__main__":
    unittest.main()
