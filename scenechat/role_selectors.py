"""Shared role-selection semantics for generated rules and restored sessions."""

ALL_ROLE_SELECTORS = {
    "*", "all", "all_active", "all_players", "all_agents", "everyone",
    "所有人", "所有角色", "所有玩家", "全部玩家", "全体玩家", "所有存活玩家",
}


def matches_role(role: str, selectors: list[str]) -> bool:
    # Eligibility (alive/active) is checked by the caller, independently of role.
    return not selectors or role in selectors or any(
        str(item).strip().lower() in ALL_ROLE_SELECTORS for item in selectors
    )


def normalize_role_selectors(selectors: list[str]) -> list[str]:
    if any(str(item).strip().lower() in ALL_ROLE_SELECTORS for item in selectors):
        return []
    return selectors


def has_phase_ending(rules, phase_name: str) -> bool:
    for rule in rules:
        if rule.phases and phase_name not in rule.phases:
            continue
        if rule.kind in {"all_of", "any_of"}:
            if has_phase_ending(rule.conditions, phase_name):
                return True
        elif rule.kind != "manual":
            return True
    return False
