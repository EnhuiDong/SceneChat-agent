"""Conservative routing for explicitly private notes in nominally public prose.

This is a literal contamination guard, not semantic declassification. Original
Markdown is preserved for the director; typed facts retain their own scopes.
Implicit secrets still require correct authoring and validation.
"""
import re
from .character_parser import split_character_profiles, _sections
from .visibility import ViewerContext, can_access


INNER_NOTE = re.compile(r'(?:实际|真实|内心)(?:的)?(?:愿望|想法|顾虑|打算|目的)|(?:心里|内心)(?:真正)?(?:希望|想要|担心|害怕)')


def split_world_context(worldview, characters, facts=None):
    """Return public prose and withheld author blocks without paid inference."""
    facts = list(facts or [])
    public_facts = [str(f.content) for f in facts if can_access(f.visibility, ViewerContext())]
    private_facts = [str(f.content) for f in facts if not can_access(f.visibility, ViewerContext())
                     and len(str(f.content).strip()) >= 8]
    names = []
    for profile in split_character_profiles(characters):
        section = _sections(profile).get(1, ('', ''))[1]
        if section.strip():
            names.append(section.splitlines()[0].strip())
    public, withheld = [], []
    for block in re.split(r'\n\s*\n', worldview):
        exact_private = any(text in block for text in private_facts)
        # Keep explicit, declared public statements. A mental-state marker
        # alone never licenses overriding an author's scoped public fact.
        inner_lines = [line.strip().lstrip('-* ').strip() for line in block.splitlines()
                       if INNER_NOTE.search(line)]
        unscoped_inner = (any(name in block for name in names) and any(
            not any(line in fact for fact in public_facts) for line in inner_lines))
        (withheld if exact_private or unscoped_inner else public).append(block)
    return '\n\n'.join(public).strip(), '\n\n'.join(withheld).strip()
