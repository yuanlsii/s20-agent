import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_runtime import (
    AgentRuntime,
    ContextManager,
    ModelResponse,
    OpenAICompatibleClient,
    Session,
    SessionStore,
    ToolCall,
    ToolRegistry,
    make_registry,
)


class ScriptedModel:
    """Deterministic model stub: no network, explicit responses, inspectable inputs."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        if not self.responses:
            raise AssertionError("model was called more times than expected")
        return self.responses.pop(0)


class EchoModel:
    def __init__(self):
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append(messages)
        user = next(message["content"] for message in reversed(messages) if message["role"] == "user")
        return ModelResponse(text=f"echo:{user}")


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class AgentRequirementsTests(unittest.TestCase):
    def make_runtime(self, root: Path, model) -> AgentRuntime:
        docs = root / "knowledge"
        docs.mkdir()
        (docs / "weekly.md").write_text("weekly report notes", encoding="utf-8")
        return AgentRuntime(
            model=model,
            registry=make_registry(docs),
            sessions=SessionStore(root / "sessions"),
        )

    def test_loop_direct_reply_has_one_model_step(self):
        model = ScriptedModel([ModelResponse(text="直接回答")])
        with tempfile.TemporaryDirectory() as folder:
            result = self.make_runtime(Path(folder), model).run("s1", "你好")
        self.assertEqual(result["answer"], "直接回答")
        self.assertEqual(len(model.calls), 1)
        self.assertNotIn("tool_call", [event["kind"] for event in result["trace"]])

    def test_progress_callback_emits_safe_status_and_final_answer(self):
        model = ScriptedModel([ModelResponse(text="直接回答")])
        events = []
        with tempfile.TemporaryDirectory() as folder:
            runtime = self.make_runtime(Path(folder), model)
            result = runtime.run(
                "stream",
                "你好",
                on_event=lambda kind, detail: events.append((kind, detail)),
            )
        self.assertEqual(result["answer"], "直接回答")
        self.assertEqual(events[-1], ("answer", "直接回答"))
        self.assertEqual([kind for kind, _ in events[:2]], ["status", "status"])
        self.assertNotIn("reasoning", json.dumps(events, ensure_ascii=False))

    def test_loop_tool_call_then_final_answer(self):
        model = ScriptedModel(
            [
                ModelResponse(
                    text="",
                    reasoning="需要计算后再回答",
                    tool_calls=[ToolCall("call-1", "calculator", {"expression": "2 + 3"})],
                ),
                ModelResponse(text="计算结果是 5"),
            ]
        )
        with tempfile.TemporaryDirectory() as folder:
            result = self.make_runtime(Path(folder), model).run("s1", "算一下 2 + 3")
        self.assertEqual(result["answer"], "计算结果是 5")
        self.assertEqual(len(model.calls), 2)
        offered_tools = {
            item["function"]["name"] for item in model.calls[0]["tools"]
        }
        self.assertIn("calculator", offered_tools)
        kinds = [event["kind"] for event in result["trace"]]
        self.assertEqual(kinds, ["user", "model", "reasoning", "tool_call", "tool_result", "model"])
        self.assertIn('"result": "5"', result["trace"][4]["detail"])

    def test_tool_registry_exposes_three_schemas_and_executes_mock_search(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            registry = make_registry(root / "knowledge")
            definitions = registry.schemas()
            names = {item["function"]["name"] for item in definitions}
            self.assertTrue({"calculator", "search", "read_docs"}.issubset(names))
            for item in definitions:
                self.assertIn("description", item["function"])
                self.assertIn("parameters", item["function"])
            (root / "knowledge").mkdir()
            (root / "knowledge" / "weather.md").write_text("weather sunny", encoding="utf-8")
            output = registry.execute(ToolCall("mock-search", "search", {"query": "weather"}))
            self.assertIn("weather.md", output)

    def test_openai_response_parser_extracts_reasoning_tools_and_final_text(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            return FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "reasoning_content": "先查文档",
                                "content": "最终答案",
                                "tool_calls": [
                                    {
                                        "id": "call-7",
                                        "function": {
                                            "name": "search",
                                            "arguments": '{"query":"S20"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            )

        client = OpenAICompatibleClient(
            "https://provider.example/v1", "secret-test-key", "test-model"
        )
        with patch("agent_runtime.urllib.request.urlopen", side_effect=fake_urlopen):
            response = client.complete([], [])
        self.assertEqual(response.reasoning, "先查文档")
        self.assertEqual(response.text, "最终答案")
        self.assertEqual(response.tool_calls[0].name, "search")
        self.assertEqual(response.tool_calls[0].arguments, {"query": "S20"})
        self.assertEqual(json.loads(captured["request"].data)["model"], "test-model")

    def test_sessions_are_isolated_and_followups_keep_context(self):
        model = EchoModel()
        with tempfile.TemporaryDirectory() as folder:
            runtime = self.make_runtime(Path(folder), model)
            first = runtime.run("window-1", "天气和待办")
            second = runtime.run("window-2", "写周报和待办")
            followup = runtime.run("window-1", "继续窗口一")
            first_session = runtime.sessions.load("window-1")
            second_session = runtime.sessions.load("window-2")
        self.assertEqual(first["answer"], "echo:天气和待办")
        self.assertEqual(second["answer"], "echo:写周报和待办")
        self.assertEqual(followup["answer"], "echo:继续窗口一")
        self.assertEqual(
            [message["content"] for message in first_session.messages if message["role"] == "user"],
            ["天气和待办", "继续窗口一"],
        )
        self.assertEqual(
            [message["content"] for message in second_session.messages if message["role"] == "user"],
            ["写周报和待办"],
        )
        self.assertNotIn("写周报和待办", json.dumps(first_session.messages, ensure_ascii=False))
        self.assertTrue(any("天气和待办" in message["content"] for message in model.calls[-1] if message["role"] == "user"))

    def test_context_compacts_old_messages_and_preserves_tool_pair(self):
        session = Session(
            id="compact",
            messages=[
                {"role": "user", "content": "old context"},
                {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
                {"role": "tool", "tool_call_id": "call-1", "content": "tool result"},
                {"role": "user", "content": "recent question"},
            ],
        )
        messages = ContextManager(recent_messages=2).build_messages(session, "system")
        self.assertTrue(session.summary)
        self.assertIn("old context", session.summary)
        roles = [message["role"] for message in messages]
        for index, role in enumerate(roles):
            if role == "tool":
                self.assertGreater(index, 0)
                self.assertEqual(roles[index - 1], "assistant")
        self.assertIn("recent question", json.dumps(messages, ensure_ascii=False))

    def test_max_steps_and_tool_errors_are_bounded(self):
        infinite_tools = ScriptedModel(
            [ModelResponse(text="", tool_calls=[ToolCall(f"call-{i}", "calculator", {"expression": "1+1"})]) for i in range(3)]
        )
        with tempfile.TemporaryDirectory() as folder:
            runtime = self.make_runtime(Path(folder), infinite_tools)
            runtime.max_steps = 2
            with self.assertRaisesRegex(RuntimeError, "maximum agent steps exceeded"):
                runtime.run("bounded", "keep going")
            unknown = runtime.registry.execute(ToolCall("bad", "missing", {}))
            self.assertIn('"ok": false', unknown)
            broken = ToolRegistry()
            broken.register("broken", "always fails", {"type": "object"}, lambda: 1 / 0)
            self.assertIn('"ok": false', broken.execute(ToolCall("x", "broken", {})))


if __name__ == "__main__":
    unittest.main()
