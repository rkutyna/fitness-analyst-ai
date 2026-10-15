"""Helpers the fact-template test files share.

`ledger` builds one ledger record per (tool, arguments) call, as the ask path
records them. `mine` picks the facts for one metric and field. `ride_ledger`
is a one-record ledger for a weekly ride_minutes impact_volume call.
"""
import json

RIDE_PERIOD = "2031-03-02:2031-03-08"


def ledger(tools, *calls):
    out = []
    for sequence, (name, arguments) in enumerate(calls, start=1):
        out.append({"sequence": sequence, "tool_name": name,
                    "arguments": arguments, "result_elided": False,
                    "result": json.loads(json.dumps(
                        getattr(tools, name)(**arguments)))})
    return out


def mine(facts, metric, field):
    return [f for f in facts.values()
            if f["metric"] == metric and f["field"] == field]


def ride_ledger() -> list[dict]:
    return [{
        "sequence": 1,
        "tool_name": "get_impact_volume",
        "arguments": {"start": "2031-03-02", "end": "2031-03-08",
                      "by": "week"},
        "result": {"metric": "ride_minutes", "period": RIDE_PERIOD,
                   "ride_minutes": 77.5, "unit": "min"},
    }]
