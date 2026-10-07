"""Lossless correction of explicitly quoted actor speech in the action field."""
import re


_SPEECH = re.compile(
    r'(?:又\s*)?(?:补(?:了|上)?(?:一句|一声|句)|回(?:了)?(?:一句|一声)|'
    r'说道|问道|回应道|开口(?:说|道)?|(?:低|扬|轻)声(?:说|道|回)|说|问|喊道)'
    r'\s*[:：]?\s*(?:“(?P<cn>[^”\n]+)”|"(?P<en>[^"\n]+)")'
)


def separate_actor_speech(action, speech, actor_name, other_names=()):
    """Move only unambiguous, actual direct speech; never infer unquoted text.

    Other speakers, reported speech, inner thoughts and hypothetical utterances
    stay untouched. This changes placement, not the words or the actor's choice.
    """
    pieces, cursor, lines = [], 0, []
    for match in _SPEECH.finditer(action):
        prefix = action[max(0, match.start() - 90):match.start()]
        clause = re.split(r"[。！？!?；;\n]", prefix)[-1]
        if any(word in clause for word in ("心想", "心里", "默念", "回忆", "记得", "记起", "想起", "听见", "听到", "读到", "写着", "本想")):
            continue
        # The verb itself belongs to the regex match: "准备说" has only
        # "准备" in the prefix. Do not turn intended/negated speech into facts.
        if re.search(r"(?:准备|打算|想要|想|正要|差点|差一点|没有|没|不曾|不肯|不愿|未|别)\s*$", clause):
            continue
        subjects = [*other_names, "对方", "旁人", "有人", "他", "她"]
        # Avoid interpreting "周衡皱着眉说…" as this actor speaking. Object
        # mentions ("看向周衡说…") may be legitimate, but ambiguous clauses
        # should be left intact instead of rewriting another person's voice.
        def other_subject(name):
            if not name or name == actor_name:
                return False
            without_objects = re.sub(
                r"(?:对着?|朝着?|向|看着|看向|望着|望向|转向)\s*" + re.escape(name), "", clause
            )
            return name in without_objects
        if any(other_subject(name) for name in subjects):
            continue
        text = match.group("cn") or match.group("en")
        pieces.append(action[cursor:match.start()])
        cursor = match.end()
        lines.append(text)
    if not lines:
        return action, speech, False
    pieces.append(action[cursor:])
    corrected_action = "".join(pieces).strip(" ，,。.;；\n") or "回应"
    corrected_speech = speech.strip()
    for line in lines:
        if line not in corrected_speech:
            corrected_speech = (corrected_speech + " " + line).strip()
    return corrected_action, corrected_speech, True
