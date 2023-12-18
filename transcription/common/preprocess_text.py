# common text preprocessing to use everywhere

import re

from num2words import num2words

# fmt: off
# regexp for weird chars found in Open Subtitles .srt files
_NON_LATIN_REGEXP = re.compile(
    "[^\\x00-\\x7F\\x80-\\xFF\\u0100-\\u017F\\u0180-\\u024F\\u1E00-\\u1EFF]"
)
_BLACKLISTED_CHARS = [
    "Ō", "Š", "Ć", "Ł", "Č", "Ñ", "Ū", "Ø", "Ã", "Ā", "Ń", "Ž", "Ă", "Å", "Ș", "Ş", "Ř", "Ý", "Ś", "Ð", "Ż", "Ę", "Đ", "Ī", "Ą", "Ě", "_", "Ğ", "Ÿ", "Ė", "Ő", "Ē", "Õ", "Ì", "Þ", "Ṣ", "o", "Ả", "Ư", "Ễ", "Ļ", "Ů", "Ĩ", "Ə", "Ľ", "Ḥ", "Ạ", "Ṭ", "Ṇ", "Ầ", "Ộ", "Ď", "Ǎ", "Ċ", "Ũ", "Ǹ", "Ʉ", "Ị", "Ồ", "Ờ", "Ű", "Ử", "Ķ", "Ề", "Ơ", "Ệ", "Ť", "Ţ", "Ŏ", "Ň", "Ņ", "Ź", "İ", "Ġ", "Ǫ", "Ẓ", "1", "2", "Ĝ", "Ħ", "Ĺ", "Ŭ", "Ų", "Ŵ", "Ɨ", "Ʋ", "ǃ", "Ǔ", "Ḍ", "Ṅ", "Ậ", "Ắ", "Ẵ", "Ế", "Ố", "Ổ", "Ớ", "Ợ", "Ụ", "Ủ", "Ứ", "Ỳ", "⁄", "Ģ", "Ț",
]
# fmt: on


def _build_unit_replacements_regexp(unit_abbrev, unit_full, fem):
    """
    ReturnBuild replacement regexps for abbreviated units.

    Case insensitive.
    Unit has to be preceded with digits and spaces.

    If there is a single '1' digit' before the unit it is replaced by
    the full form as-is, otherwise an 's' is happened to the full form since it is plural.

    For the singlar case we also replace the leading '1' by 'un' or 'une'
    (because this won't be handled properly by num2words)
    """

    singular_replacement = "une" if fem else "un"
    return [
        (
            re.compile(rf"(?<=\b)1 *{unit_abbrev}(?=(\d|\b))", re.I),
            f" {singular_replacement} {unit_full} ",
        ),
        (re.compile(rf"(?<=\d) *{unit_abbrev}s?(?=(\d|\b))", re.I), f" {unit_full}s "),
    ]


def _build_title_replacement_regexp(title_abbrev, title_full, dot):
    """
    Replace an abbreviate title by its expanded form, optionally
    taking into account a trailing dot
    """

    if dot:
        title_abbrev += r"\.?"
    return (re.compile(rf"\b{title_abbrev}\b"), f" {title_full} ")


# replacements regexps to run on each utterance groundtruth text
# order matters, they have to be run before everything else, and in this order!
_REPLACEMENT_REGEXPS = (
    [
        # expand abbreviated titles
        _build_title_replacement_regexp("Mme", "Madame", dot=False),
        _build_title_replacement_regexp("Mlle", "Mademoiselle", dot=False),
        _build_title_replacement_regexp("Mr", "Monsieur", dot=True),
        _build_title_replacement_regexp("Pr", "Professeur", dot=True),
        _build_title_replacement_regexp("Dr", "Docteur", dot=True),
        # special case for M / M., we have to be more careful as it could be in an accronym
        (re.compile(r"( M\. |^M\. | M\.$)"), " Monsieur "),
        (re.compile(r"(^| )M[\. ](?=[a-zA-Z]{3})"), " Monsieur "),
        # misc special chars
        (re.compile(r"\+"), " plus "),
        (re.compile(r"\&"), " et "),
        (re.compile(r"%"), " pourcent "),
        (re.compile(r"(?<=\b)n ?°(?=[^a-zA-Z])", re.I), " numéro "),
        # special case for 1st/2d ordinals
        # (other numbers are handled with num2words)
        (re.compile(r"(?<=\b)1 ?er(?=\b)", re.I), " premier "),
        (re.compile(r"(?<=\b)1 ?[eè]re(?=\b)", re.I), " première "),
        (re.compile(r"(?<=\b)2de(?=\b)", re.I), " seconde "),
        (re.compile(r"(?<=\b)2d(?=\b)", re.I), " second "),
        # spaces/dots in numbers (ex: 100.000 or 100 000)
        (re.compile(r"(?<=\d) (?=\d\d\d)"), ""),
        (re.compile(r"(?<=0)\.(?=000)"), ""),
    ]
    # expand abbreviated units
    + _build_unit_replacements_regexp("h", "heure", fem=True)
    + _build_unit_replacements_regexp("min", "minute", fem=True)
    + _build_unit_replacements_regexp("sec", "seconde", fem=True)
    + _build_unit_replacements_regexp("km", "kilomètre", fem=False)
    + _build_unit_replacements_regexp("kg", "kilo", fem=False)
    + _build_unit_replacements_regexp("°", "degré", fem=False)
    + [
        # special case for km/h
        (re.compile(r"km [/ ]{0,3}h"), "kilomètres heure"),
    ]
    # misc oral stuff
    + [
        (re.compile(r"\beuh\b", re.I), " "),
        (re.compile(r"\bheu\b", re.I), " "),
        (re.compile(r"\bhm\b", re.I), " "),
        (re.compile(r"\bhmm\b", re.I), " "),
        (re.compile(r"\bhmmm\b", re.I), " "),
        (re.compile(r"\bbah\b", re.I), " ben "),
    ]
)

_ORDINALS_REGEXP = re.compile(r"(\d+) ?[eè](me)?\b")
_CARDINAL_NUMBERS = re.compile(r"\d+([\.,]\d*)?")
_SPECIAL_CHARS_REGEXP = re.compile(r"[^\w ]")
_DUPL_WHITESPACE_REGEXP = re.compile(r" +")
_PARTIAL_WORDS_WITH_SLASHES_REGEXP = re.compile(r"(/\w+)|(\w+/)")


def preprocess_text(text, drop_slashes=False):
    """
    Preprocess a groundtruth utterance text so it is ready to be tokenized.

    This is used:
    - before "training" the tokenizer
    - before training a language model
    - before traning an ASR models

    What is done:
    - expand common abbreviations
    - replace digits
    - remove special chars
    - convet to uppperase

    If drop_slashes is True, incomplete words begining or ending with "/" will
    be removed (used for pxslu).

    Will return None if text is left with no meaningful characters after pre-processing
    (or if it contains weirds non-ascii characters that would otherwise pollute tokenization)
    """

    # run all basic replacement regexps
    for regexp, replacement in _REPLACEMENT_REGEXPS:
        text = regexp.sub(replacement, text)

    if drop_slashes:
        text = _PARTIAL_WORDS_WITH_SLASHES_REGEXP.sub("", text)

    # replace numbers with num2word
    # ordinals (1ère, 2ème, etc)
    numbers_to_replace = [
        (m.span(0), num2words(m.group(1), lang="fr", to="ordinal"))
        for m in _ORDINALS_REGEXP.finditer(text)
    ]
    for span, number in numbers_to_replace[::-1]:
        text = text[: span[0]] + " " + number + " " + text[span[1] :]
    # cardinals (1, 2, etc)
    numbers_to_replace = [
        (m.span(0), num2words(m.group(0).replace(",", "."), lang="fr"))
        for m in _CARDINAL_NUMBERS.finditer(text)
    ]
    for span, number in numbers_to_replace[::-1]:
        text = text[: span[0]] + " " + number + " " + text[span[1] :]

    # replace special chars/punctuations by whitespace
    text = _SPECIAL_CHARS_REGEXP.sub(" ", text)
    # remove duplicate whitespace and strip
    text = _DUPL_WHITESPACE_REGEXP.sub(" ", text)
    text = text.strip()

    # uppercase
    text = text.upper()

    if len(text) == 0:
        return None

    # discard if contains weird chars (useful for .srts from opensubtitles)
    # we want to get rid of them otherwise they will use some "slots" in the set of tokens
    # of the tokenizer, if we ask for 100% character coverage
    if _NON_LATIN_REGEXP.search(text) is not None or any(
        c in text for c in _BLACKLISTED_CHARS
    ):
        return None

    return text


_WER_REPLACEMENT_REGEXPS = [
    (re.compile(r"\bAH\b"), "HA"),
    (re.compile(r"\bOH\b"), "HO"),
    (re.compile(r"\bEH\b"), "HE"),
    (re.compile(r"\bT ES\b"), "TU ES"),
    (re.compile(r"\bT AS\b"), "TU AS"),
    (re.compile(r"\bY A\b"), "IL Y A"),
    (re.compile("Œ"), "OE"),
    (re.compile("Æ"), "AE"),
]


def prepare_words_for_wer(text):
    """
    Perform additional replacements before WER computing, and return text as list of words.

    We don't do these replacements in preprocess_text() because we would stil like
    our models to learn them but they are not very meaningful when computing WER.
    (ex: Œ vsOE, "TU ES" vs "T ES", etc)
    """ 
    text = text.upper()
    for regexp, replacement in _WER_REPLACEMENT_REGEXPS:
        text = regexp.sub(replacement, text)
    # FIXME not sure why this is needed, there shouldn't be any "'" anyway
    text = text.replace("'", " ")
    words = text.split(" ")
    return words
