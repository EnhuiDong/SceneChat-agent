"""Keep explicitly unknown author notes out of an actor's model input.

Knowing that a dossier says "does not know X" is still seeing X. This
projection removes the value of labelled nonknowledge fields, not the saved
canon or actual observations. It is structural, not a semantic secret detector.
"""
import re


UNKNOWN = re.compile(
    r'^[ \t]*(?:[-*][ \t]+)?(?:\*\*)?'
    r'(?:确定不知道|不知道|未知(?:信息|事项|内容)?|尚未知道|尚未掌握|未掌握|unknown|not_known)'
    r'(?:\*\*)?[ \t]*[:：]', re.IGNORECASE)
FIELD = re.compile(
    r'^[ \t]*(?:[-*][ \t]+)?(?:\*\*)?'
    r'(?:确定知道|已知|知道|怀疑|推测|可能错误相信|不确定|错误相信|确定不知道)'
    r'(?:\*\*)?[ \t]*[:：]')
OMITTED = '尚未掌握的作者资料不提供具体内容；不能据此补造事实。'


def project_nonknowledge(text):
    """Hide values (including continuation lines) of explicit unknown fields.

    Keep the generic lack-of-knowledge marker, later known/suspected fields and
    numbered dossier sections. Apply to each memory record separately so a
    multiline initial note cannot swallow an unrelated later memory.
    """
    lines, withheld = [], False
    for line in str(text or '').splitlines():
        if line.startswith('我的初始知识边界：'):
            lines.append('我的初始知识边界：')
            line = line[len('我的初始知识边界：'):]
        start = UNKNOWN.match(line)
        if start:
            lines.append(line[:start.end()] + OMITTED)
            withheld = True
        elif re.match(r'^\s*#{1,6}\s', line) or FIELD.match(line):
            withheld = False
            lines.append(line)
        elif not withheld:
            lines.append(line)
        elif not line.strip():
            lines.append(line)
    return '\n'.join(lines)
