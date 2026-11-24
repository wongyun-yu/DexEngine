"""Legacy stored text — values the code wrote in Korean before the English names. A reader that decides on a stored
value refuses such a value instead of reading it as something else; a legacy migration (not shipped)
has to rewrite it first. No imports: the state
vocabulary (placement.ledger_states) reads it."""

#: Hangul syllables — the old names' script (no English name holds one).
HANGUL_FIRST, HANGUL_LAST = chr(0xAC00), chr(0xD7A3)
MIGRATE_HINT = "migrate the legacy values over the data root (no migration tool is shipped)"


def is_legacy(text: str) -> bool:
    """True when the text holds a Hangul syllable."""
    return any(HANGUL_FIRST <= ch <= HANGUL_LAST for ch in text)
