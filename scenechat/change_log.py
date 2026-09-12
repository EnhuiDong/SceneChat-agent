"""Committed state differences; never model-generated explanations or intentions."""
from copy import deepcopy


def snapshot(state):
    return deepcopy({"public": state.public_world_state(), "director": {
        "世界": state.world_state, "实体": state.entity_states,
        **{a.name: {"目标": a.goal_status, "关系": a.relationship_dynamics,
                  "承诺": a.pending_commitments, "认知": [vars(b) for b in a.belief_records],
                  "位置": a.current_location, "可行动": a.eligible, "资源": a.resources}
           for a in state.agents.values()}}})


def differences(before, after, event_id):
    result = []
    def walk(a, b, path):
        if a == b:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(a.keys() | b.keys()):
                walk(a.get(key), b.get(key), path + [key])
        else:
            result.append({"path": " / ".join(path), "before": a, "after": b, "event_id": event_id})
    walk(before, after, [])
    return result
