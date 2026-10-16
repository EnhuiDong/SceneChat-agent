"""Evidence-backed scene opportunities, not a second director or fact writer.

Only actor-visible receipts and executable interfaces are projected. Milestone
prose, future outcomes and another character's private state never enter this
path. A suggestion does not commit a decision or resolve an obligation.
"""
from dataclasses import dataclass
import json

from .config import config_value
from .visibility import ViewerContext, can_access


ORDINARY_ACTIONS = {"speak", "act", "observe", "pass"}
CAUSAL_OPERATIONS = {
    "set_world", "increment_world", "set_entity", "move_agent",
    "set_agent_status", "set_resource", "consume_resource", "record_vote",
    "add_known_fact", "set_phase", "set_goal_status",
}


def changed_operations(state, operations):
    """Called BEFORE commit: receipts, not affect or an ability-use counter."""
    from .mechanics import patch_changes_state
    return [index for index, op in enumerate(operations)
            if op.get("op") in CAUSAL_OPERATIONS and patch_changes_state(state, [op])]


def _visible_change(state, agent, message):
    if agent.name in message.individual_observations:
        return False  # Do not mine the original patch behind altered perception.
    viewer = ViewerContext(name=agent.name, role=agent.role, location=agent.current_location)
    # Legacy traces have only a whole-patch flag: retain it conservatively,
    # but still filter the original operations by their visibility.
    indexes = message.intent.get("causal_progress_operations")
    if not isinstance(indexes, list):
        indexes = range(len(message.state_patch)) if message.intent.get("meaningful_state_change") else []
    for index in indexes:
        if type(index) is not int or not 0 <= index < len(message.state_patch):
            continue
        op = message.state_patch[index]
        kind = op.get("op")
        if kind in {"set_world", "increment_world"}:
            field = state.state_schema.get(op.get("key"))
            if field and can_access(field.visibility, viewer):
                return True
        elif kind == "set_entity":
            entity = getattr(state.world_spec, "entities", {}).get(op.get("target"))
            if isinstance(entity, dict) and can_access(entity.get("visibility"), viewer):
                return True
        elif kind in {"move_agent", "record_vote", "set_phase", "set_agent_status"}:
            return True
        elif kind in {"add_known_fact", "set_resource", "consume_resource", "set_goal_status"}:
            if op.get("target") == agent.name:
                return True
    return False


def visible_stall(state, agent):
    """Local actor turns without a committed visible change; not a quality score.

    Open conversation can legitimately have no engine changes. This heuristic
    ONLY offers choices, never flags a generation failure or forces an ending.
    """
    turns, anchor = 0, "opening"
    for message in reversed(state.history[-80:]):
        if not state._agent_can_observe(agent, message):
            continue
        if message.authoritative and _visible_change(state, agent, message):
            anchor = message.event_id
            break
        if message.speaker in state.agents and not message.intent.get("generation_fallback"):
            turns += 1
    return turns, anchor


@dataclass(frozen=True)
class SceneOpportunity:
    id: str
    kind: str
    anchor: str
    source_event_ids: tuple[str, ...] = ()
    rule_id: str = ""
    action_type: str = ""
    target: str = ""
    ability: str = ""

    def receipt(self):
        from dataclasses import asdict
        return asdict(self)


def _attempted(state, agent, opportunity):
    def metadata(message, key):
        value = message.intent.get(key)
        return value if isinstance(value, dict) else {}

    return any(m.speaker == agent.name and not m.intent.get("generation_fallback")
               and metadata(m, "scene_opportunity").get("id") == opportunity.id
               and metadata(m, "scene_opportunity").get("anchor") == opportunity.anchor
               and (opportunity.kind != "interface" or state.turn_count - m.turn < 4
                    or metadata(m, "scene_step").get("choice") == "decline")
               for m in state.history[-80:])


def _targets(state, actor, rule):
    from .mechanics import visible_entities
    scope = rule.target_scope
    if scope == "none":
        return [""]
    if scope == "self":
        return [actor.name]
    if scope in {"object", "proposal"}:
        return [key for key, item in visible_entities(state, actor).items() if item.get("kind") == scope][:4]
    if scope == "location":
        return list(state.locations)[:4]
    if scope in {"same_location", "character", "any_active"}:
        # Never offer a hidden, distant character by name.
        return [a.name for a in state.agents.values()
                if a.eligible and a.current_location == actor.current_location][:4]
    return []


def _interface_opportunities(state, actor, anchor):
    from .role_selectors import matches_role
    from .runtime import Intent, IntentResolver
    from .mechanics import _visible_rule_effects
    viewer = ViewerContext(name=actor.name, role=actor.role, location=actor.current_location)
    resolver = IntentResolver()
    phase = state.phase_specs.get(state.current_phase)
    if getattr(state.world_spec, "execution_version", 1) != 2:
        return  # No predictions of implicit legacy tabletop outcomes.
    def mainline_rank(rule):
        # Rank only public executable condition footprints, never director
        # milestone prose or secret prerequisites/completion information.
        public = ViewerContext()
        from .mechanics import visible_entities
        visible = visible_entities(state, actor)
        score = 0
        for beat in getattr(state.world_spec, 'beat_specs', []):
            if beat.id in state.arc_state.resolved_beat_ids or beat.prerequisites:
                continue
            if beat.phase_hint and beat.phase_hint != state.current_phase:
                continue
            for item in beat.completion_conditions:
                kind = item.get('kind')
                if kind == 'world_equals':
                    field = state.state_schema.get(item.get('key'))
                    if not field or not can_access(field.visibility, public):
                        continue
                    match = any(e.op == 'set_world' and e.key == item.get('key')
                                and type(e.value) is type(item.get('value')) and e.value == item.get('value')
                                for e in rule.effects)
                elif kind == 'entity_equals':
                    entity = getattr(state.world_spec, 'entities', {}).get(item.get('target'), {})
                    if (item.get('target') not in visible or not isinstance(entity, dict)
                            or not can_access(entity.get('visibility'), public)):
                        continue
                    match = any(e.op == 'set_entity' and e.target == item.get('target')
                                and e.key == item.get('key') and type(e.value) is type(item.get('value'))
                                and e.value == item.get('value') for e in rule.effects)
                else:
                    continue
                score += bool(match)
        return score

    for rule in sorted(state.rules[:40], key=mainline_rank, reverse=True):
        if (rule.action_type in ORDINARY_ACTIONS
                or rule.phases and state.current_phase not in rule.phases
                or phase and phase.allowed_action_types and rule.action_type not in phase.allowed_action_types
                or not matches_role(actor.role, rule.allowed_roles)
                or not can_access(rule.visibility, viewer)):
            continue
        # Filtering on an undisclosed prerequisite could leak its truth merely
        # by making an opportunity appear/disappear. Do not query those rules.
        from .action_requirements import visible_requirement_items
        if len(visible_requirement_items(state, actor, rule)) != len(rule.requires):
            continue
        if any(item.get("kind") == "all_active_at_location" for item in rule.requires):
            continue  # Globally queried presence is not necessarily observable.
        mode = rule.effect_mode if getattr(state.world_spec, "execution_version", 1) == 2 else "stack"
        abilities = [a for a in actor.ability_states.values()
                     if a.action_type in {rule.action_type, "act"} and a.available
                     and (not a.phases or state.current_phase in a.phases)][:4]
        choices = abilities if mode == "ability" else [None, *abilities] if mode == "stack" else [None]
        for ability in choices:
            for target in _targets(state, actor, rule):
                intent = Intent(actor=actor.name, action_type=rule.action_type,
                                action="", rule_id=rule.id, target=target, ability=ability.id if ability else "")
                result = resolver.resolve(state, intent)
                if not result.accepted or not changed_operations(state, result.patch.operations):
                    continue
                from types import SimpleNamespace
                receipt = SimpleNamespace(individual_observations={},
                    state_patch=result.patch.operations,
                    intent={"causal_progress_operations": changed_operations(state, result.patch.operations)})
                if not _visible_change(state, actor, receipt):
                    continue
                # Offer only visible rule outcomes. Knowledge/revelation
                # abilities and hidden costs are not scored as progress hints.
                if not _visible_rule_effects(state, actor, rule) and mode == "rule":
                    continue
                if any(op.get("op") == "set_phase" for op in result.patch.operations):
                    continue  # Existing phase-exit scheduling owns this route.
                if not any(op.get("op") in {"set_entity", "move_agent", "record_vote", "set_world"}
                           for op in result.patch.operations):
                    continue
                opportunity = SceneOpportunity(f"interface:{rule.id}:{intent.ability}:{target}",
                                               "interface", anchor, rule_id=rule.id,
                                               action_type=rule.action_type, target=target,
                                               ability=intent.ability)
                if not _attempted(state, actor, opportunity):
                    yield opportunity


def scene_opportunity(state, actor):
    if config_value("simulation", "scene_progression", "off") != "opportunities" or not actor.eligible:
        return None
    phase = state.phase_specs.get(state.current_phase)
    if phase and (phase.event_only or phase.allowed_action_types
                  and not set(phase.allowed_action_types).intersection(ORDINARY_ACTIONS)):
        return None  # Balloting/action-only phases already have a clear task.
    from .pacing import PacingPolicy
    turns, anchor = visible_stall(state, actor)
    threshold = max(PacingPolicy.from_value(state.arc_state.pace).stagnation_limit,
                    len(state.agents) * (phase.opening_min_cycles if phase else 1))
    if turns < threshold:
        return None
    from .agenda import visible_to
    tasks = [task for task in state.agenda.values()
             if task.phase == state.current_phase and task.status in {"completed", "blocked"}
             and task.owner == actor.name and visible_to(state, task, actor)]
    # A new answer deserves to be heard; its requester can face the choice.
    latest = next((m for m in reversed(state.history)
                   if state._agent_can_observe(actor, m) and not m.intent.get("generation_fallback")), None)
    if latest and latest.intent.get("conversation_move") == "reveal" and not any(
            latest.event_id in task.evidence_event_ids for task in tasks):
        return None
    interface = next(_interface_opportunities(state, actor, anchor), None)
    if interface:
        return interface
    for task in sorted(tasks, key=lambda t: t.updated_at_turn, reverse=True):
        ids = tuple(event_id for event_id in task.evidence_event_ids
                    if any(m.event_id == event_id and state._agent_can_observe(actor, m)
                           and not m.intent.get("generation_fallback") for m in state.history))
        opportunity = SceneOpportunity(f"response:{task.id}", "response", anchor, ids[-4:])
        if ids and not _attempted(state, actor, opportunity):
            return opportunity
    return None


def progression_context(state, actor):
    opportunity = scene_opportunity(state, actor)
    if opportunity is None:
        return ""
    if opportunity.kind == "interface":
        detail = json.dumps({"rule_id": opportunity.rule_id, "action_type": opportunity.action_type,
                             "target": opportunity.target, "ability": opportunity.ability}, ensure_ascii=False)
        rule = next(r for r in state.rules if r.id == opportunity.rule_id)
        from .mechanics import _visible_rule_effects
        outcome = json.dumps(_visible_rule_effects(state, actor, rule), ensure_ascii=False)
        deferred = []
        for message in state.history[-80:]:
            prior = message.intent.get('scene_opportunity')
            choice = message.intent.get('scene_step')
            if (message.speaker == actor.name and not message.intent.get('generation_fallback')
                    and isinstance(prior, dict) and isinstance(choice, dict)
                    and prior.get('id') == opportunity.id and prior.get('anchor') == opportunity.anchor
                    and choice.get('choice') == 'defer'):
                deferred.append({'event_id': message.event_id,
                                 'reason': str(choice.get('because', ''))[:240],
                                 'action_quote': message.action[:320]})
        followthrough = ("\n【本人已延后过的原文记录】" + json.dumps(deferred[-2:], ensure_ascii=False)
                         + "。这些理由不是新增事实。若仍处理同一件事，这次把本人能独立完成的连续一段做到自然边界，"
                         "不只准备姿势、递一件又等一轮，也不为了继续延后新增物品、工序或障碍。"
                         "实际还有对方的选择、未知结果或规则步骤时，保留该边界，不代办、不保证成功。"
                         if deferred else "")
        return ("【局面中的可行下一步——机会，不是命令或已经发生的结果】\n"
                "可见局面已有一段时间没有实际变化。当前接口允许你尝试：" + detail
                + "；此动作的已声明用途：" + rule.description
                + "；若依法提交才可能产生的可见登记：" + outcome
                + "。其含义与前提见合法动作清单；没有替你决定愿不愿做，也不保证成功。"
                "先在本次完整 Intent JSON 中填写私有 scene_step："
                '{"choice":"take|defer|decline","because":"这次实际选择的一个具体原因"}，再写公开 speech/action 和执行字段。'
                "take 表示本次真的尝试此接口；defer 须指出眼前仍未完成的具体步骤，"
                "并在本次言行里处理它，不重复准备、确认或已经完成的动作；decline 表示本人不愿做，"
                "让这一拒绝在实际交流中留下边界或代价，不用反问把决定权交回同一个循环。"
                "按你的牵挂、性格和对眼前人的态度决定是否做、拒绝或提出不同条件；"
                "若做就交真实接口，若不做让人知道这次选择带来的具体障碍，不原样重讲整场论证。"
                "若本人已经作出与此用途对应的实际选择，这次用该 action_type/rule_id 提交，"
                "不用先只以 speak 宣布打算、下轮又讨论是否开始。表演演到自己的选择边界，不预写他人反应。"
                "无需宣布正在推进剧情，不用新反问替代自己的选择，不替他人做决定。" + followthrough)
    return ("【已有回应后的选择机会——不要求和解或改变性格】\n"
            "你之前提出的事项已收到回应，来源：" + ",".join(opportunity.source_event_ids)
            + "。收到回应不等于相信或解决，但对方不是还没有回答。"
            "这次可以落实自己能做的一步、说清接受什么条件、改变争取方式、拒绝并暂时搁置；"
            "不要只重问同一个保证。继续坚持也应面对已经收到的回答和真实代价。"
            "没有新证据时不编背景、外部回应或他人的承诺来推动；选择和语气由完整人设决定。")


def step_receipt(raw, opportunity, intent):
    """Private diagnostic/opt-out only: no fact, task or state authority."""
    if not isinstance(raw, dict) or opportunity is None:
        return {}
    choice, because = raw.get('choice'), raw.get('because')
    if choice not in {'take', 'defer', 'decline'} or not isinstance(because, str):
        return {}
    if choice == 'take' and (intent.action_type != opportunity.action_type
                            or intent.rule_id != opportunity.rule_id
                            or intent.target != opportunity.target
                            or intent.ability != opportunity.ability):
        return {}  # Saying take does not substitute for the actual attempt.
    return {'choice': choice, 'because': because[:240]}
