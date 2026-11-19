"""Turn one Wikibooks Cookbook page (raw wikitext) into a structured recipe.

We parse wikitext ourselves (regex + brace matching) instead of rendering
HTML: the infobox fields, section headings and categories are all explicit
in the markup, and the markup we need to strip is a small, known set.

    parse_recipe(title, wikitext) -> {
        "title", "category", "difficulty", "servings", "time_text",
        "total_minutes", "categories": [...],
        "sections": [{"section": "Ingredients", "text": "..."}, ...]
    }
"""

from __future__ import annotations

import html
import re

# ------------------------------------------------------------ templates

def _find_template(text: str, names: tuple[str, ...]) -> tuple[int, int] | None:
    """Span of the first {{name|...}} template (brace-matched), or None."""
    pat = re.compile(r"\{\{\s*(" + "|".join(names) + r")\s*[|}]", re.I)
    m = pat.search(text)
    if not m:
        return None
    depth, i = 0, m.start()
    while i < len(text) - 1:
        if text.startswith("{{", i):
            depth += 1
            i += 2
            continue
        if text.startswith("}}", i):
            depth -= 1
            i += 2
            if depth == 0:
                return m.start(), i
            continue
        i += 1
    return m.start(), len(text)


def _split_params(body: str) -> dict[str, str]:
    """Split a template body on top-level '|' into name=value params."""
    parts, depth, cur, i = [], 0, [], 0
    while i < len(body):
        two = body[i:i + 2]
        if two in ("{{", "[["):
            depth += 1; cur.append(two); i += 2; continue
        if two in ("}}", "]]"):
            depth -= 1; cur.append(two); i += 2; continue
        if body[i] == "|" and depth == 0:
            parts.append("".join(cur)); cur = []; i += 1; continue
        cur.append(body[i]); i += 1
    parts.append("".join(cur))
    params = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            params[k.strip().lower()] = v.strip()
    return params


INFOBOX_NAMES = ("recipe ?summary", "recipesummary")


def parse_infobox(wikitext: str) -> dict[str, str]:
    span = _find_template(wikitext, INFOBOX_NAMES)
    if not span:
        return {}
    return _split_params(wikitext[span[0] + 2:span[1] - 2])


# ------------------------------------------------------------ cleaning

def _convert(m: re.Match) -> str:
    """{{convert|8|oz|g|abbr=on}} -> '8 oz'; {{convert|1|-|2|cups}} -> '1-2 cups'."""
    args = [a.strip() for a in m.group(1).split("|") if "=" not in a]
    if len(args) >= 4 and args[1] in ("-", "to", "–", "and"):
        return f"{args[0]}–{args[2]} {args[3]}"
    return " ".join(args[:2])


def _innermost_templates(text: str) -> str:
    """Strip templates inside-out, keeping the readable ones' text."""
    text = re.sub(r"\{\{\s*(?:br|clear|-)\s*\}\}", "; ", text, flags=re.I)
    for _ in range(10):
        new = re.sub(r"\{\{\s*convert\s*\|([^{}]*)\}\}", _convert, text, flags=re.I)
        new = re.sub(r"\{\{\s*(?:frac)\s*\|([^{}|]*)\|([^{}|]*)(?:\|([^{}|]*))?\}\}",
                     lambda m: (f"{m.group(1)} {m.group(2)}/{m.group(3)}" if m.group(3)
                                else f"{m.group(1)}/{m.group(2)}"), new, flags=re.I)
        new = re.sub(r"\{\{[^{}]*\}\}", "", new)
        if new == text:
            break
        text = new
    return text


def _link(m: re.Match) -> str:
    inner = m.group(1)
    target, _, label = inner.partition("|")
    t = target.strip()
    if re.match(r"(?i)(file|image|category|media):", t):
        return ""
    if label:
        return label.split("|")[-1]
    return re.sub(r"(?i)^:?cookbook:\s*", "", t)


def clean(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    # gallery lines are "File:x.jpg|caption": keep the caption (often a step)
    text = re.sub(r"<gallery[^>]*>(.*?)</gallery>",
                  lambda m: "\n".join(l.split("|", 1)[1] for l in m.group(1).splitlines()
                                      if "|" in l),
                  text, flags=re.S | re.I)
    text = _innermost_templates(text)
    # [[File:...|...[[nested]]...]] — drop file/image links with nesting
    for _ in range(3):
        text = re.sub(r"\[\[(?:File|Image):(?:[^\[\]]|\[\[[^\]]*\]\])*\]\]", "", text, flags=re.I)
    text = re.sub(r"\[\[([^\[\]]*)\]\]", _link, text)
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)
    text = re.sub(r"<\s*/?\s*br\s*/?\s*>", "; ", text, flags=re.I)
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)
    text = re.sub(r"'''?", "", text)
    text = re.sub(r"__[A-Z]+__", "", text)
    text = html.unescape(text)
    lines = []
    n = 0
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("{|") or s.startswith("|") or s.startswith("!"):
            s = s.lstrip("{|!}-").strip()        # flatten wiki tables crudely
        if re.match(r"^#+", s):
            n += 1
            s = f"{n}. " + s.lstrip("#*: ").strip()
        else:
            if not s.startswith("*"):
                n = 0 if s else n
            if s.startswith("*"):
                s = "- " + s.lstrip("*: ").strip()
        s = re.sub(r"[ \t]+", " ", s)
        lines.append(s)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ------------------------------------------------------------ time

_UNIT_MIN = {"min": 1, "minute": 1, "minutes": 1, "mins": 1, "m": 1,
             "hour": 60, "hours": 60, "hr": 60, "hrs": 60, "h": 60,
             "day": 1440, "days": 1440}
_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "half an": 0.5, "a half": 0.5, "an": 1, "a": 1}


def parse_minutes(time_text: str) -> int | None:
    """Total minutes from free text like 'Prep: 20 minutes; Baking: 1 hour'.
    Ranges ('20–30 minutes') count their upper end, so a 'under N minutes'
    filter never admits a recipe that might take longer. None if unparseable."""
    if not time_text:
        return None
    t = time_text.lower().replace("½", ".5").replace("¼", ".25").replace("¾", ".75")
    for w, n in _WORD_NUM.items():
        t = re.sub(rf"\b{w}\b(?=\s+(?:min|hour|hr|day))", str(n), t)
    # "Prep: 5 min; Cooking: 10 min; Total: 15 min" -> trust the Total, don't sum all three
    m_total = re.search(r"total\s*:?(.*)$", t)
    if m_total and re.search(r"\d", m_total.group(1)):
        t = m_total.group(1)
    total, found = 0.0, False
    pat = r"(\d+(?:\.\d+)?)(?:\s*(?:-|–|to)\s*(\d+(?:\.\d+)?))?\s*(minutes?|mins?|hours?|hrs?|days?|h|m)\b"
    for m in re.finditer(pat, t):
        val = float(m.group(2) or m.group(1))
        total += val * _UNIT_MIN[m.group(3)]
        found = True
    return int(round(total)) if found else None


# ------------------------------------------------------------ recipe

def parse_recipe(title: str, wikitext: str) -> dict:
    box = parse_infobox(wikitext)
    categories = sorted(set(c.strip() for c in
                            re.findall(r"\[\[\s*Category\s*:\s*([^\]|]+)", wikitext, re.I)))
    body = wikitext
    span = _find_template(body, INFOBOX_NAMES)
    if span:
        body = body[:span[0]] + body[span[1]:]
    body = re.sub(r"\[\[\s*Category\s*:[^\]]*\]\]", "", body, flags=re.I)

    sections, current, buf = [], "Introduction", []
    for line in body.splitlines():
        m = re.match(r"^\s*(=+)\s*(.*?)\s*\1\s*$", line)
        if m:
            sections.append((current, "\n".join(buf)))
            current, buf = clean(m.group(2)) or "Untitled", []
        else:
            buf.append(line)
    sections.append((current, "\n".join(buf)))

    out_sections = []
    for name, raw in sections:
        txt = clean(raw)
        if txt and not re.fullmatch(r"[\s;|:,.-]*", txt):
            out_sections.append({"section": name, "text": txt})

    def field(k):
        return clean(box.get(k, "")) or None

    time_text = field("time")
    diff = re.search(r"\d", box.get("difficulty", ""))
    name = re.sub(r"^Cookbook:", "", title)
    return {
        "title": name,
        "category": field("category"),
        "difficulty": int(diff.group()) if diff else None,
        "servings": field("servings") or field("yield"),
        "time_text": time_text,
        "total_minutes": parse_minutes(time_text or ""),
        "categories": categories,
        "sections": out_sections,
    }
