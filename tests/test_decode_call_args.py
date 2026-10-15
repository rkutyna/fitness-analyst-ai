"""The one decode of a proposed tool call's arguments, shared by both loops (#102).

The loops' behaviour on undecodable arguments is pinned end to end in
test_tool_args_decode.py; this pins the helper's contract directly.
"""
import pytest

from health_advisor import llm


@pytest.mark.parametrize("call, expected", [
    ({"function": {"name": "t", "arguments": {"a": 1}}}, ({"a": 1}, False)),
    ({"function": {"name": "t", "arguments": '{"a": 1}'}}, ({"a": 1}, False)),
    ({"function": {"name": "t"}}, ({}, False)),
    ({"function": {"name": "t", "arguments": ""}}, ({}, False)),
    ({"function": {"name": "t", "arguments": None}}, ({}, False)),
    ({}, ({}, False)),
    ({"function": {"name": "t", "arguments": "{not json"}}, ({}, True)),
    ({"function": {"name": "t", "arguments": '{"a": '}}, ({}, True)),
    # Decodable but not a mapping: handed back as decoded, not flagged.
    ({"function": {"name": "t", "arguments": "[1, 2]"}}, ([1, 2], False)),
])
def test_decode_call_args(call, expected):
    assert llm._decode_call_args(call) == expected
