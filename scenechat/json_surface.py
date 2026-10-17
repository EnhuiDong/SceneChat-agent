"""Repair JSON notation outside strings without rewriting story content.

No missing values, closing brackets, quotes or fields are invented. Python
literal compatibility remains limited to JSON-shaped data, never eval/code.
"""
import ast
import json
import math
import re


def _segments(text):
    """Yield (is_quoted, text), retaining escaped single/double quoted values."""
    start, index = 0, 0
    while index < len(text):
        if text[index] not in ('"', "'"):
            index += 1
            continue
        if start < index:
            yield False, text[start:index]
        start, quote = index, text[index]
        index += 1
        while index < len(text):
            if text[index] == '\\':
                index += 2
            elif text[index] == quote:
                index += 1
                break
            else:
                index += 1
        yield True, text[start:index]
        start = index
    if start < len(text):
        yield False, text[start:]


def _strip_comments(text):
    result, index = [], 0
    while index < len(text):
        char = text[index]
        if char in ('"', "'"):
            start, quote = index, char
            index += 1
            while index < len(text):
                if text[index] == '\\':
                    index += 2
                elif text[index] == quote:
                    index += 1
                    break
                else:
                    index += 1
            result.append(text[start:index])
        elif text.startswith('//', index):
            end = text.find('\n', index + 2)
            index = len(text) if end < 0 else end
            result.append(' ')
        elif text.startswith('/*', index):
            end = text.find('*/', index + 2)
            if end < 0:
                raise ValueError('Unclosed JSON comment')
            result.append(' ')
            index = end + 2
        else:
            result.append(char)
            index += 1
    return ''.join(result)


def _json_shape(value):
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_json_shape(item) for item in value)
    return type(value) is dict and all(type(k) is str and _json_shape(v) for k,v in value.items())


def decode_surface(candidate):
    repaired = []
    for quoted, part in _segments(_strip_comments(candidate)):
        if not quoted:
            part = re.sub(r',\s*([}\]])', r'\1', part)
            part = re.sub(r'([,{]\s*)([A-Za-z_\u4e00-\u9fff][\w\-\u4e00-\u9fff]*)(\s*:)',
                          r'\1"\2"\3', part)
        repaired.append(part)
    text = ''.join(repaired)
    try:
        payload, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        # Only bare JSON constants are translated for literal_eval; the word
        # "true" inside dialogue, URLs and regexes is never changed.
        python = ''.join(part if quoted else re.sub(r'\b(true|false|null)\b',
                         lambda m: {'true':'True','false':'False','null':'None'}[m.group()], part)
                         for quoted, part in _segments(text))
        payload = ast.literal_eval(python)
    if not _json_shape(payload):
        raise ValueError('Compatibility output must contain JSON-shaped values only')
    return payload
