"""Closed authoring constraints, separate from diegetic knowledge.

Only code-owned instructions reach performers. A hidden brief's excerpts,
identities, ending and director prose are never copied into an actor prompt.
These instructions do not verify facts or grant new knowledge.
"""

POLICIES = {
    "no_new_amounts": "不增加未设定的具体金额或数额；不把未知的薪水、费用、账款临时编成事实。",
    "no_new_cast_members": "不增加新的登场人物，不替未声明的人说话、行动或走进当前场景；这不禁止讨论设定中已有的背景人物。",
    "no_new_hidden_defects": "不凭空添加未设定的隐藏缺陷、故障、损坏或隐瞒的责任；真实已发生的新变化仍可承接。",
    "no_new_task_items": "不扩张已明确限定的交接、任务或挑战事项，不临时增加新的必办项目；这不限制人物自己的情感与顾虑。",
    "no_new_background_people": "不补未设定的联系人、同事、亲友或其他背景人物；不知道谁负责时可以直说不知道，不能编一个名字来回答。",
    "no_new_background_items": "不补未设定的物品、文件、凭证、设备或已有记录；可以提出之后要制作或取得它们，不能说它们已经存在。",
    "no_new_background_quantities": "不补未设定的金额、比例、日期、地址、号码和次数；可以约定新的计划，但不能把临时猜出的数字说成既有事实。",
    "no_new_background_history": "不补未设定的往事、约定、通知、已完成的手续或此前联络；承接实际已发生的事件，不临时编出过去来支撑答复。",
}


def normalize_codes(value):
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(code for code in value if isinstance(code, str) and code in POLICIES))


def parse_policies(value):
    if not isinstance(value, list):
        return []
    result = []
    seen = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        code, source = item.get("code"), item.get("source_excerpt")
        if (not isinstance(code, str) or code not in POLICIES or code in seen
                or not isinstance(source, str) or not source.strip()):
            continue
        result.append({"code": code, "source_excerpt": source.strip()})
        seen.add(code)
    return result


def render_policy(world):
    codes = normalize_codes(getattr(world, "performance_policies", []))
    if not codes:
        return ""
    return ("【用户明确指定的创作边界——不是人物获知的剧情事实】\n"
            "以下约束只限定作者如何补写，不透露任何隐藏身份、真相或未来情节。"
            "人物仍可以有感受、误解、不确定的推测和自己的新打算，不必变成拒答机器。\n"
            + "\n".join("- " + POLICIES[code] for code in codes))
