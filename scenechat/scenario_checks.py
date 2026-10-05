"""Cross-field checks for generated casts and executable phase exits."""

import re

from .role_selectors import ALL_ROLE_SELECTORS, has_phase_ending, matches_role


def _stated_faction_count(text: str, faction: str) -> int | None:
    numerals = "零一二两三四五六七八九十"
    matches = list(re.finditer(
        rf"([0-9{numerals}]+)\s*(?:名|个|位)?\s*(?:是|为|属于)?\s*{re.escape(faction)}",
        text, flags=re.IGNORECASE,
    ))
    if not matches:
        return None
    value = matches[-1].group(1)
    if value.isdigit():
        return int(value)
    digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if value == "十":
        return 10
    if "十" in value:
        left, right = value.split("十", 1)
        return (digits.get(left, 1) if left else 1) * 10 + digits.get(right, 0)
    return digits.get(value)


def rule_actors(package, rule, phase):
    if phase.event_only or (rule.phases and phase.name not in rule.phases):
        return []
    if phase.allowed_action_types and rule.action_type not in phase.allowed_action_types:
        return []
    return [
        actor for actor in package.characters
        if matches_role(actor.role, phase.actor_roles)
        and matches_role(actor.role, rule.allowed_roles)
    ]


def additional_scenario_issues(package, user_prompt=None):
    from .mechanics import validate_mechanics
    issues = validate_mechanics(package)
    factions = {actor.faction for actor in package.characters if actor.faction}
    def check_winner(rule):
        if rule.kind == "faction_parity" and rule.winner in factions and rule.winner != rule.faction:
            issues.append(f"结束规则 {rule.id} 的 faction_parity 阵营方向与 winner 相反：faction 是达到人数优势的一方，应与获胜阵营一致")
        for child in rule.conditions:
            check_winner(child)
    for rule in package.world.termination_rules:
        check_winner(rule)
    # Only user text may authorize a title as the name, never generated assumptions.
    source = user_prompt if user_prompt is not None else "\n".join(
        item.source_excerpt or item.content for item in package.brief.constraints
    )
    locked_cast_text = "\n".join(
        item.content for item in package.brief.constraints
        if item.locked and item.category in {"cast", "rule"}
    )
    for faction in factions:
        expected = _stated_faction_count(source, faction)
        if expected is None:
            expected = _stated_faction_count(locked_cast_text, faction)
        if expected is not None:
            actual = sum(actor.faction == faction for actor in package.characters)
            if actual != expected:
                issues.append(
                    f"用户硬约束指定阵营“{faction}”{expected}名，实际生成{actual}名；"
                    "请同步修正角色 faction、私有身份和关联秘密，不得改变用户指定的人数"
                )
    explicit_title_style = bool(re.search(
        r"(?:用|以|按|使用).{0,8}(?:职业|职务|称谓|称呼).{0,6}(?:称呼|命名|姓名|名字|代称)|不用姓名|不使用姓名",
        source,
    ))
    title_pattern = re.compile(
        r"[\u4e00-\u9fff]{0,2}(?:医生|律师|教授|探长|警官|黑客|作家|记者|工程师|老师|护士|队长)"
    )
    for actor in package.characters:
        if title_pattern.fullmatch(actor.name) and actor.name not in source and not explicit_title_style:
            issues.append(
                f"角色姓名“{actor.name}”使用了未经用户指定的职业称呼；请生成独立姓名，"
                "职业保留在 public_identity/role，并同步所有关系键、事实 scope 和导演笔记中的姓名引用"
            )
        if any(token in package.world.public_world_markdown for token in ("隐藏身份", "身份隐藏", "秘密身份", "伪装")):
            public_profile = " ".join((actor.public_identity, actor.public_traits, actor.public_background))
            if re.search(r"(?:实际是|其实是|真实身份是|隐藏身份是|秘密身份是|实际上是)", public_profile):
                issues.append(f"角色“{actor.name}”的公开档案包含隐藏身份说明；应只保留其他角色可知的背景")

    roles = {actor.role for actor in package.characters}
    for kind, specs, field in (
        ("阶段", package.world.phase_specs, "actor_roles"),
        ("规则", package.world.rules, "allowed_roles"),
    ):
        for spec in specs:
            unknown = [
                role for role in getattr(spec, field)
                if role not in roles and role.strip().lower() not in ALL_ROLE_SELECTORS
            ]
            if unknown:
                issues.append(
                    f"{kind}“{getattr(spec, 'name', getattr(spec, 'id', ''))}”引用不存在的角色职能："
                    f"{', '.join(unknown)}；不得依赖角色名单外的 host，所有可行动角色使用空角色列表"
                )

    phases = {phase.name: phase for phase in package.world.phase_specs}
    for phase in phases.values():
        if phase.advance_when not in {"manual", "after_event", "all_eligible_acted", "all_active_voted"}:
            issues.append(f"阶段“{phase.name}”使用未知 advance_when")
        if not phase.event_only and not any(
            matches_role(actor.role, phase.actor_roles) for actor in package.characters
        ):
            issues.append(f"阶段“{phase.name}”没有任何符合 actor_roles 的角色")
        if phase.event_only:
            terminal = not phase.next_phase and has_phase_ending(package.world.termination_rules, phase.name)
            if not terminal and (phase.advance_when != "after_event" or not phase.next_phase):
                issues.append(f"环境事件阶段“{phase.name}”必须使用 after_event 并明确 next_phase；无后续阶段时必须有适用的结构化结束规则，不能无限旁白")
            seen = set()
            current = phase
            while current is not None and current.event_only:
                if current.name in seen:
                    issues.append(f"环境事件阶段“{phase.name}”形成没有角色行动出口的循环")
                    break
                seen.add(current.name)
                current = phases.get(current.next_phase)
        elif phase.advance_when == "after_event":
            issues.append(f"角色阶段“{phase.name}”不能依赖 after_event 退出，请使用角色行动完成条件")

    for rule in package.world.rules:
        selected_phases = [phases[name] for name in rule.phases if name in phases]
        if selected_phases and all(phase.event_only for phase in selected_phases):
            issues.append(
                f"规则“{rule.id}”只绑定环境事件阶段，角色永远无法执行；"
                "请将结算放到可执行的角色规则或由环境事件呈现已提交结果"
            )
        if phases and not any(rule_actors(package, rule, phase) for phase in phases.values()):
            issues.append(f"规则“{rule.id}”没有可执行的角色和阶段组合，请同时核对角色职能与 allowed_action_types")
        if package.world.execution_version == 1 and rule.action_type == "inspect" and re.search(r"提问|质疑|询问|拷问", rule.description) and not re.search(
            r"查验|读取|揭示|获取.{0,8}(?:身份|阵营)", rule.description
        ):
            issues.append(f"规则“{rule.id}”把普通提问设为 inspect，会意外获得目标阵营；请使用 speak 的 question/challenge")

    effects = [effect for rule in package.world.rules for effect in rule.effects]
    effects.extend(effect for actor in package.characters for ability in actor.abilities for effect in ability.effects)
    written_keys = {effect.key for effect in effects if effect.op in {"set_world", "increment_world"}}

    def check_end(rule):
        if rule.kind == "world_equals" and rule.key in package.world.initial_state:
            initial = package.world.initial_state[rule.key]
            if isinstance(initial, (int, bool)) and type(rule.value) is not type(initial):
                issues.append(f"结束规则“{rule.id}”的 world_equals.value 与状态“{rule.key}”类型不一致；不能把 >= 等运算符当作目标值")
            field = package.world.state_schema.get(rule.key)
            # Public state can also be updated by validated narrator events.
            # Private counters, however, need an explicit producer; the public
            # narrator cannot maintain hidden faction counts on their behalf.
            if (
                initial != rule.value and rule.key not in written_keys
                and field is not None and "public" not in field.visibility
                and field.mutable_by == ["resolver"]
            ):
                issues.append(f"结束规则“{rule.id}”依赖从未被规则更新的状态“{rule.key}”；人数胜负应使用 faction_eliminated/faction_parity 查询实际存活角色")
        for child in rule.conditions:
            check_end(child)

    for rule in package.world.termination_rules:
        check_end(rule)
    return issues
