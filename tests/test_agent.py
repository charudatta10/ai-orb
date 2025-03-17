import unittest
from unittest.mock import MagicMock
from ai_orb.agent import Think
import ollama
from ai_orb.agent import Act, SecureSandbox

class TestThink(unittest.TestCase):
    def setUp(self):
        # Mock the LLM client
        self.mock_llm = MagicMock(spec=ollama.Client)
        self.tools = {
            "sample_tool": lambda x: f"Processed {x}"
        }
        self.think = Think(llm=self.mock_llm, tools=self.tools)

    def test_generate_plan_success(self):
        # Mock the LLM response
        mock_response = {
            'response': """
            ```json
            [
                {"action": "step1", "tool": "sample_tool", "input": {"data": "test"}, "expected_output": "result1"},
                {"action": "step2", "tool": "sample_tool", "input": {"data": "test2"}, "expected_output": "result2"}
            ]
            ```
            """
        }
        self.mock_llm.generate.return_value = mock_response

        goal = "Test Goal"
        context = {"key": "value"}

        plan = self.think.generate_plan(goal, context)

        expected_plan = [
            {"action": "step1", "tool": "sample_tool", "input": {"data": "test"}, "expected_output": "result1"},
            {"action": "step2", "tool": "sample_tool", "input": {"data": "test2"}, "expected_output": "result2"}
        ]

        self.assertEqual(plan, expected_plan)

    def test_generate_plan_invalid_response(self):
        # Mock the LLM response with invalid format
        mock_response = {'invalid_key': 'invalid_value'}
        self.mock_llm.generate.return_value = mock_response

        goal = "Test Goal"
        context = {"key": "value"}

        plan = self.think.generate_plan(goal, context)

        self.assertEqual(plan, [])

    def test_generate_plan_exception(self):
        # Mock the LLM to raise an exception
        self.mock_llm.generate.side_effect = Exception("LLM error")

        goal = "Test Goal"
        context = {"key": "value"}

        plan = self.think.generate_plan(goal, context)

        self.assertEqual(plan, [])

        class TestAct(unittest.TestCase):
            def setUp(self):
                # Mock the SecureSandbox
                self.mock_sandbox = MagicMock(spec=SecureSandbox)
                self.act = Act(sandbox=self.mock_sandbox)

            def test_execute_tool_success(self):
                # Mock the sandbox execution to return a successful result
                self.mock_sandbox.execute.return_value = "Execution result"

                tool_name = "sample_tool"
                input_data = {"key": "value"}

                result = self.act.execute_tool(tool_name, input_data)

                expected_result = {"success": True, "output": "Execution result"}
                self.assertEqual(result, expected_result)

            def test_execute_tool_failure(self):
                # Mock the sandbox execution to raise an exception
                self.mock_sandbox.execute.side_effect = Exception("Execution error")

                tool_name = "sample_tool"
                input_data = {"key": "value"}

                result = self.act.execute_tool(tool_name, input_data)

                expected_result = {"success": False, "error": "Execution error"}
                self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()