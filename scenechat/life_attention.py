"""Sparse invitations to broaden an actor's attention, never new story facts.

The receipt lives on an accepted event, so checkpoint restore preserves it.
Visible engine stagnation is only an opportunity heuristic, not a literary
quality verdict: a conversation need not change a world variable to matter.
"""
from .progression import ORDINARY_ACTIONS, visible_stall
from .prompt_library import prompt_contract


def attention_offer(state, actor):
    phase = state.phase_specs.get(state.current_phase)
    if (not actor.eligible or state.last_scheduler_decision.get("phase_exit_rule_id")
            or phase and (phase.event_only or phase.allowed_action_types
                          and not set(phase.allowed_action_types).intersection(ORDINARY_ACTIONS))):
        return None
    for pending in actor.pending_intents:
        try:
            urgency = float(pending.get("urgency", 0))
        except (TypeError, ValueError):
            urgency = 0
        if urgency >= 0.75:
            return None  # An urgent request is not an invitation to change subject.
    turns, anchor = visible_stall(state, actor)
    if turns < max(4, len(state.agents)):
        return None
    own_events = [message for message in state.history
                  if message.speaker == actor.name and not message.intent.get("generation_fallback")
                  and state._agent_can_observe(actor, message)]
    if not own_events:
        return None
    for message in own_events:
        receipt = message.intent.get("life_attention")
        if isinstance(receipt, dict) and receipt.get("anchor") == anchor:
            return None  # Offered once; the actor may ignore it, without punishment.
    return {"anchor": anchor, "offered": True}


def attention_context(state, actor):
    if attention_offer(state, actor) is None:
        return ""
    return ("\n【可以打开的一点生活余地，不是本轮必做任务】\n"
            + prompt_contract("life_exchange"))
