"""Require committed evidence for narrator claims about controlled outcomes."""
import re
from .dialogue_quality import DialogueQualityIssue

NUMBER = r"[0-9零〇一二两三四五六七八九十百]+"


def _actor_claim_pattern(state, name, tail, distance=18):
    """Do not attach another subject's outcome across a sentence or name.

    "林默获得一票。周屿被淘汰" contains no claim that 林默 was eliminated.
    Commas remain allowed for terse readbacks such as "周屿，六票，被淘汰".
    """
    others = "|".join(re.escape(other) for other in state.agents if other != name)
    boundary = rf"(?!{others})" if others else ""
    return re.escape(name) + rf"(?:{boundary}[^。！？!?；;\n]){{0,{distance}}}" + tail


def _number(value):
    if value.isdigit():
        return int(value)
    digits = {character: index for index, character in enumerate("零一二三四五六七八九")}
    digits.update({"〇": 0, "两": 2})
    total, current = 0, 0
    for character in value:
        if character in {"十", "百"}:
            total += (current or 1) * (10 if character == "十" else 100)
            current = 0
        else:
            current = digits.get(character, 0)
    return total + current


def _latest_vote_batch(messages, actor_names):
    """Replay committed ballots; the last cast vote alone is not a tally."""
    ballots, latest, closing = {}, {}, ""
    sources, latest_sources = set(), set()
    eligible = set(actor_names)
    for message in messages:
        for op in message.state_patch:
            if op.get("op") == "set_agent_status" and op.get("key") in {"alive", "active"}:
                if op.get("value") is False:
                    eligible.discard(op.get("target"))
            if op.get("op") == "record_vote":
                ballots[op.get("actor")] = op.get("target")
                sources.add(message.event_id)
                latest_sources = set(sources)
                latest, closing = dict(ballots), ""
            elif op.get("op") == "settle_votes" and eligible and eligible.issubset(ballots):
                latest, closing = dict(ballots), message.event_id
            elif op.get("op") == "clear_votes" and ballots:
                latest, closing = dict(ballots), message.event_id
                ballots = {}
                sources = set()
            elif op.get("op") == "set_phase":
                if ballots:
                    latest, closing = dict(ballots), message.event_id
                ballots = {}
                sources = set()
    return latest, closing, latest_sources


def fresh_vote_readback(state, parsed):
    """A verified new ballot result is not repeated scenery or an old result.

    This exemption only concerns textual similarity. Evidence, tally, cast,
    secrecy and phase checks still run independently and must all pass.
    """
    text = "。".join(part for part in re.split(r"[。！？；\n]", str(parsed.get("narration") or ""))
                    if not re.search(r"如果|假如|尚未|并未|没有|不会|可能|不是|并非", part))
    if not re.search(r"投票(?:结果|结束)|平票|(?:获得|得|获|以)\s*" + NUMBER + r"\s*票", text):
        return False
    if inspect_claims(state, parsed):
        return False
    known = [m for m in state.history if m.authoritative and m.kind != "narration"]
    ballots, closing, _ = _latest_vote_batch(known, state.agents)
    if not ballots or not closing or closing not in parsed.get("evidence_event_ids", []):
        return False
    return not any(m.kind == "narration" and closing in m.intent.get("evidence_event_ids", [])
                   for m in state.history)


def inspect_claims(state, parsed):
    # Do not interpret explicitly hypothetical/negative clauses as committed facts.
    raw_text = str(parsed["narration"] or "")
    text = "。".join(part for part in re.split(r"[。！？；\n]", raw_text)
                    if not re.search(r"如果|假如|尚未|并未|没有|不会|可能", part))
    ids = parsed.get("evidence_event_ids", [])
    known = {m.event_id: m for m in state.history if m.authoritative and m.kind != "narration"}
    issues = []
    phase = state.phase_specs.get(state.current_phase)
    alive_count = sum(agent.eligible for agent in state.agents.values())
    for count in re.finditer(
        r"(?P<marker>其余|剩余|还剩|仍有|现有|存活)\s*(" + NUMBER + r")\s*(?:名|个|位)?\s*(?:玩家|角色|人)",
        raw_text,
    ):
        clause = re.split(r"[。！？；\n]", raw_text[:count.start()])[-1]
        if re.search(r"上一轮|前一轮|当时|此前|曾经|过去", clause):
            continue
        stated = count.group(2)
        if _number(stated) != alive_count:
            issues.append(DialogueQualityIssue(
                "wrong_alive_count",
                f"当前只有{alive_count}名可行动角色；若指全体存活者，不能写成{stated}人。",
                hard=count.group("marker") != "其余",
            ))
    if phase is not None and phase.advance_when == "all_active_voted" and state.votes:
        if re.search(r"(?:本轮|当前).{0,12}(?:无有效结果|平票|已结算|结算完成|投票结束)|(?:投票|票数|统计).{0,8}(?:清空|归零|重置)", raw_text):
            issues.append(DialogueQualityIssue(
                "premature_vote_settlement",
                "当前仍有选票待投，不能宣布本轮无效、平票、已结算或清空票数。",
                hard=True,
            ))
        if re.search(r"游戏未开始|投票未开始", raw_text):
            issues.append(DialogueQualityIssue(
                "contradictory_game_state",
                "已有真实选票，本场游戏与投票不可能仍处于未开始状态。",
                hard=True,
            ))
    allowed_phases = {state.current_phase}
    if phase and (phase.event_only or phase.advance_when == "after_event"):
        allowed_phases.add(phase.next_phase)
    for name in state.phase_sequence:
        if name not in allowed_phases and re.search(r"(?:进入|开启|开始|切换至).{0,6}" + re.escape(name), text):
            issues.append(DialogueQualityIssue("unexecuted_phase_change", f"引擎仍在{state.current_phase}，不能宣布进入{name}。", hard=True))
    if re.search(r"(?:进入|开启|开始|转入).{0,6}(?:加投|重投|重新投票)|(?:加投|重投)(?:环节|阶段)|请.{0,8}重新(?:投票|选择)", text):
        if not any("投票" in name or "vote" in name.lower() for name in allowed_phases):
            issues.append(DialogueQualityIssue(
                "unsupported_revote", "当前规则没有进入加投阶段；平票结果不能由旁白擅自改成重新投票。", hard=True,
            ))
    current_round = state.world_state.get("round")
    ongoing_round = re.search(r"第(" + NUMBER + r")轮.{0,10}(?:仍在|正在|尚未).{0,8}(?:投票|讨论|进行)", text)
    if type(current_round) is int and ongoing_round and _number(ongoing_round.group(1)) != current_round:
        issues.append(DialogueQualityIssue(
            "wrong_round_reference", f"当前是第{current_round}轮，不能把较早轮次写成仍在进行。", hard=True,
        ))
    if any(not isinstance(i, str) or i not in known for i in ids):
        issues.append(DialogueQualityIssue("unknown_outcome_evidence", "结果引用必须来自已提交角色或干预事件，不能引用旁白或虚构 ID。", hard=True))
    evidence = [known[i] for i in ids if isinstance(i, str) and i in known
                and (parsed.get("visibility") == "audience_only" or "public" in known[i].scopes)]
    vote_claim = bool(re.search(r"(?:平票|[获得以]\s*" + NUMBER + r"\s*票|选票.{0,8}(?:汇入|统计)|投票(?:结果|结束|通道.{0,4}关闭))", text))
    if vote_claim and not any(op.get("op") in {"record_vote", "settle_votes"} for m in evidence for op in m.state_patch):
        issues.append(DialogueQualityIssue("unexecuted_vote_result", "没有引用真实投票事件，不得宣布票数、平票或投票完成。若尚未投票，将行动留给角色。", hard=True))
    elif vote_claim:
        ballots, closing, sources = _latest_vote_batch(known.values(), state.agents)
        counts = {name: list(ballots.values()).count(name) for name in set(ballots.values())}
        valid = bool(closing and closing in {m.event_id for m in evidence})
        if parsed.get("visibility") != "audience_only":
            valid = valid and all("public" in known[source].scopes for source in sources)
        if "平票" in text:
            valid = valid and bool(counts) and list(counts.values()).count(max(counts.values())) > 1
        for name in state.agents:
            match = re.search(_actor_claim_pattern(state, name, r"(?:获得|得|获|以)\s*(" + NUMBER + r")\s*票", 8), text)
            match = match or re.search(re.escape(name) + r"\s*[，,:：]\s*(" + NUMBER + r")\s*票", text)
            if match and counts.get(name, 0) != _number(match.group(1)):
                valid = False
            for target in state.agents:
                if re.search(re.escape(name) + r"(?:自己)?投给" + re.escape(target), text) and ballots.get(name) != target:
                    valid = False
        if not valid:
            issues.append(DialogueQualityIssue("unsupported_vote_tally", "票数或平票结论与最近已完成投票不符，或没有引用该轮结算事件。请只呈现已有公开结果。", hard=True))
    for name in state.agents:
        result_pattern = _actor_claim_pattern(state, name, r"(?:被[^。！？!?；;\n]{0,10}(?:淘汰|带离|隔离)|已出局|已离场)", 12)
        if re.search(result_pattern, text):
            evidence_ids = {item for item in ids if isinstance(item, str)}
            if evidence_ids and any(
                previous.kind == "narration"
                and evidence_ids == set(previous.intent.get("evidence_event_ids") or [])
                and re.search(result_pattern, previous.speech)
                for previous in state.history[-16:]
            ):
                issues.append(DialogueQualityIssue(
                    "repeated_outcome_announcement",
                    f"{name}的离场已经播报过；相同证据不能再次播报同一结果。", hard=True,
                ))
        if re.search(_actor_claim_pattern(state, name, r"身份[^。！？!?；;\n]{0,12}(?:已|被)[^。！？!?；;\n]{0,6}(?:公开|揭晓|公布)", 8), text):
            reveal = any(op.get("op") == "set_world" and re.search(r"身份|揭晓|公开|identity|reveal", str(op.get("key", "")), re.I)
                         for m in evidence for op in m.state_patch)
            if not reveal:
                issues.append(DialogueQualityIssue("unexecuted_identity_reveal", f"没有{name}身份公开的执行证据；出局不自动等于身份被公开。", hard=True))
        if re.search(_actor_claim_pattern(state, name, r"(?:被[^。！？!?；;\n]{0,12}(?:淘汰|拖离|隔离|杀死)|已经死亡|已出局)"), text):
            isolation = bool(re.search(_actor_claim_pattern(state, name, r"被[^。！？!?；;\n]{0,12}(?:拖离|隔离)"), text))
            if not any(op.get("target") == name and (isolation and op.get("op") == "move_agent" or
                       op.get("op") == "set_agent_status" and op.get("key") in {"alive", "active"} and op.get("value") is False)
                       for m in evidence for op in m.state_patch):
                issues.append(DialogueQualityIssue("unexecuted_cast_result", f"{name}没有对应的已提交离场/伤亡证据；不能用旁白替代结算。", hard=True))
    return issues
