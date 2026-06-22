#!/usr/bin/env python3
"""Readable-text extractor for Mathematica 5.2 box-format ``.nb`` notebooks.

The notebooks shipped with Honeychurch's *Simulating Electrochemical Reactions
in Mathematica* (SERM) store every cell as a nested ``Cell[<content>, "Style",
opts...]`` expression, where ``<content>`` is itself a box-expression
(``BoxData[RowBox[...]]`` for code, ``TextData[...]`` for prose).  There is no
Wolfram kernel available here, so this module recovers a *readable* rendering of
the original Wolfram source (for code cells) and plain text (for prose cells) so
a human or an LLM can study and re-implement the algorithms in Python.

Design goals (and the bugs they fix relative to the first cut):

* **Correct cell boundaries.**  Cells are parsed with a bracket-aware,
  string-aware scanner over the *top-level* ``Notebook[{ ... }]`` cell list, so
  cells nested inside option values (e.g. page-header ``Cell[...]`` blobs) are
  *not* mistaken for real content cells.
* **Correct style detection.**  A cell's style is the first *top-level* bare
  string argument after the content argument -- not "the first known style
  token found anywhere in the body" (which mis-classified Title cells whose
  options embed nested cells of other styles).
* **Clean prose.**  Only the content argument is rendered, and structural style
  markers / option strings are never concatenated into the text.
* **Dropped blobs.**  ``Output``, ``Print``, ``Graphics`` and ``Message`` cells,
  and any cell carrying a cached ``GraphicsData``/``PostScript`` blob, are
  dropped.

Usage::

    python nb_extract.py path/to/Notebook.nb [-o out.txt]

With no ``-o`` the readable text is written to stdout.
"""
from __future__ import annotations

import argparse
import re
import sys

# --- cell styles ----------------------------------------------------------
PROSE_STYLES = {
    "Title", "Subtitle", "Subsubtitle", "Section", "Subsection",
    "Subsubsection", "Text", "Caption", "Item", "ItemNumbered",
    "Abstract", "Initialize", "Author", "Affiliation",
}
CODE_STYLES = {"Input", "InputOnly", "InitializationCell", "Code", "Program"}
DROP_STYLES = {
    "Output", "Print", "Graphics", "Message", "Copyright", "PageNumber",
    "Header", "Footer",
}
# Styles whose content is pure presentation boilerplate we never want.
SKIP_PROSE_STYLES = {"Copyright"}


# --- named-character de-sugaring ------------------------------------------
NAMED_CHAR = {
    r"\[Rule]": "->",
    r"\[RuleDelayed]": ":>",
    r"\[Equal]": "==",
    r"\[NotEqual]": "!=",
    r"\[LessEqual]": "<=",
    r"\[GreaterEqual]": ">=",
    r"\[Alpha]": "alpha",
    r"\[Beta]": "beta",
    r"\[Gamma]": "gamma",
    r"\[Delta]": "delta",
    r"\[CapitalDelta]": "Delta",
    r"\[Epsilon]": "epsilon",
    r"\[Lambda]": "lambda",
    r"\[Mu]": "mu",
    r"\[Nu]": "nu",
    r"\[Pi]": "Pi",
    r"\[Tau]": "tau",
    r"\[Theta]": "theta",
    r"\[Phi]": "phi",
    r"\[CurlyPhi]": "phi",
    r"\[Psi]": "psi",
    r"\[Omega]": "omega",
    r"\[CapitalOmega]": "Omega",
    r"\[Sigma]": "sigma",
    r"\[Rho]": "rho",
    r"\[Eta]": "eta",
    r"\[Zeta]": "zeta",
    r"\[Xi]": "xi",
    r"\[Kappa]": "kappa",
    r"\[Infinity]": "Infinity",
    r"\[PartialD]": "D",
    r"\[Element]": " in ",
    r"\[Times]": "*",
    r"\[Divide]": "/",
    r"\[Cross]": "cross",
    r"\[LeftArrow]": "<-",
    r"\[RightArrow]": "->",
    r"\[RightArrowLeftArrow]": "<=>",
    r"\[LeftRightArrow]": "<=>",
    r"\[LongRightArrow]": "->",
    r"\[Equilibrium]": "<=>",
    r"\[Function]": "&",
    r"\[IndentingNewLine]": "\n",
    r"\[LineSeparator]": "\n",
    r"\[NonBreakingSpace]": " ",
    r"\[Prime]": "'",
    r"\[Hyphen]": "-",
    r"\[Dash]": "-",
    r"\[LongDash]": "--",
    r"\[CenterDot]": ".",
    r"\[Degree]": "deg",
    r"\[Sqrt]": "Sqrt",
    r"\[Copyright]": "(c)",
    r"\[RegisteredTrademark]": "(R)",
    r"\[DiscretionaryHyphen]": "",
    r"\[Bullet]": "*",
    r"\[Ellipsis]": "...",
    r"\[CloseCurlyQuote]": "'",
    r"\[OpenCurlyQuote]": "'",
    r"\[CloseCurlyDoubleQuote]": '"',
    r"\[OpenCurlyDoubleQuote]": '"',
    r"\[RawDoubleQuote]": '"',
    r"\[DoubleStruckCapitalD]": "DM",
    r"\[LeftDoubleBracket]": "[[",
    r"\[RightDoubleBracket]": "]]",
    r"\[LeftSkeleton]": "<<",
    r"\[RightSkeleton]": ">>",
}


def desugar_named_chars(s: str) -> str:
    for k, v in NAMED_CHAR.items():
        s = s.replace(k, v)
    # any remaining \[Something] -> Something
    s = re.sub(r"\\\[([A-Za-z]+)\]", r"\1", s)
    return s


def unescape_wl_string(s: str) -> str:
    r"""Decode a raw WL string-literal body to its text.

    Handles ``\n``, ``\t``, ``\"``, ``\\`` and the 5.2 ``\<...\>`` "raw string"
    delimiters (which we simply strip).
    """
    s = s.replace("\\<", "").replace("\\>", "")
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt == "n":
                out.append("\n"); i += 2; continue
            if nxt == "t":
                out.append("\t"); i += 2; continue
            if nxt == '"':
                out.append('"'); i += 2; continue
            if nxt == "\\":
                out.append("\\"); i += 2; continue
            if nxt == "\n":  # line continuation
                i += 2; continue
        out.append(c)
        i += 1
    return "".join(out)


# --- top-level expression scanning ----------------------------------------
def split_top_level(body: str):
    """Split a comma-separated argument string into top-level pieces.

    Bracket-, brace- and string-aware: commas inside ``[]``/``{}`` or string
    literals do not split.  Returns the list of trimmed top-level argument
    strings.
    """
    parts = []
    depth = 0
    in_str = False
    start = 0
    i = 0
    n = len(body)
    while i < n:
        c = body[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c in "[{":
                depth += 1
            elif c in "]}":
                depth -= 1
            elif c == "," and depth == 0:
                parts.append(body[start:i].strip())
                start = i + 1
        i += 1
    tail = body[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _match_bracket(text: str, start: int) -> int:
    """Return index just past the ``]`` matching the ``[`` at ``start-1``.

    ``start`` points just inside the opening bracket.  String- and brace-aware.
    """
    depth = 1
    j = start
    n = len(text)
    in_str = False
    while j < n and depth > 0:
        c = text[j]
        if in_str:
            if c == "\\":
                j += 2
                continue
            if c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c in "[{":
                depth += 1
            elif c in "]}":
                depth -= 1
        j += 1
    return j  # index just past the matching close bracket


def iter_cells(text: str):
    """Yield the body of every content ``Cell[...]``, recursing into groups.

    The notebook is a tree: ``Cell[CellGroupData[{Cell[...], Cell[...]}], ...]``.
    We descend the *content* argument of each cell only.  When a cell's content
    is a ``CellGroupData[{...}]`` we recurse into the contained cell list; we
    never scan a cell's *option* arguments, so page-header/footer ``Cell[...]``
    blobs embedded in options are skipped.
    """
    def find_cells(s: str):
        """Yield bodies of ``Cell[...]`` that start at top level of ``s``."""
        i = 0
        n = len(s)
        while i < n:
            m = re.compile(r"\bCell\[").search(s, i)
            if not m:
                return
            inner = m.end()
            end = _match_bracket(s, inner)
            body = s[inner:end - 1]
            yield body
            i = end

    def walk(s: str):
        for body in find_cells(s):
            args = split_top_level(body)
            if not args:
                continue
            content = args[0].strip()
            if content.startswith("CellGroupData["):
                # Recurse into the group's cell list (its first argument).
                gstart = content.index("[") + 1
                gend = _match_bracket(content, gstart)
                yield from walk(content[gstart:gend - 1])
            else:
                yield body

    yield from walk(text)


def cell_content_and_style(body: str):
    """Return ``(content_arg, style)`` for a cell body ``content, "Style", ...``.

    ``style`` is the first *top-level* bare-string argument after the content
    argument; ``None`` if no recognisable style is present.
    """
    args = split_top_level(body)
    if not args:
        return None, None
    content = args[0]
    style = None
    for a in args[1:]:
        a = a.strip()
        if a.startswith('"') and a.endswith('"'):
            inner = unescape_wl_string(a[1:-1]).strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", inner):
                style = inner
                break
    return content, style


# --- box-expression flattener (for code cells) ----------------------------
def tokenize(src: str):
    """Tokenize a box-expression into (kind, value) tuples."""
    i, n = 0, len(src)
    toks = []
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
            continue
        if c == '"':
            j = i + 1
            buf = []
            while j < n:
                if src[j] == "\\" and j + 1 < n:
                    buf.append(src[j]); buf.append(src[j + 1]); j += 2; continue
                if src[j] == '"':
                    break
                buf.append(src[j]); j += 1
            toks.append(("str", "".join(buf)))
            i = j + 1
            continue
        if c in "[]{},":
            toks.append(("punct", c))
            i += 1
            continue
        m = re.match(r"[A-Za-z$][A-Za-z0-9$`]*", src[i:])
        if m:
            toks.append(("head", m.group(0)))
            i += m.end()
            continue
        toks.append(("punct", c))
        i += 1
    return toks


# Box heads handled specially; "*Box" wrappers we don't know are stripped to
# their first child, and everything else renders as head[args].
_STRIP_BOX = {
    "StyleBox", "FormBox", "AdjustmentBox", "TagBox", "InterpretationBox",
    "AnimatorBox", "ButtonBox", "FrameBox", "PaneBox", "TooltipBox",
    "ItemBox", "DynamicBox",
}


def flatten_boxes(toks, pos=0):
    """Return ``(text, new_pos)`` rendering one box-expression to source text."""
    tok = toks[pos]
    kind, val = tok
    if kind == "str":
        return unescape_wl_string(val), pos + 1
    if kind == "head":
        if pos + 1 < len(toks) and toks[pos + 1] == ("punct", "["):
            args, pos = _parse_bracket(toks, pos + 2)
            return render_box(val, args), pos
        return val, pos + 1
    if tok == ("punct", "{"):
        items, pos = _parse_brace(toks, pos + 1)
        return "{" + ", ".join(items) + "}", pos
    return val, pos + 1


def _parse_bracket(toks, pos):
    args = []
    while pos < len(toks):
        t = toks[pos]
        if t == ("punct", "]"):
            return args, pos + 1
        if t == ("punct", ","):
            pos += 1
            continue
        text, pos = flatten_boxes(toks, pos)
        args.append(text)
    return args, pos


def _parse_brace(toks, pos):
    items = []
    while pos < len(toks):
        t = toks[pos]
        if t == ("punct", "}"):
            return items, pos + 1
        if t == ("punct", ","):
            pos += 1
            continue
        text, pos = flatten_boxes(toks, pos)
        items.append(text)
    return items, pos


def render_box(head, args):
    """Render a known box head; fall back to ``head[args]``."""
    if head == "RowBox":
        # RowBox[{a, b, c}] concatenates its literal children.
        if len(args) == 1 and args[0].startswith("{") and args[0].endswith("}"):
            inner = args[0][1:-1]
            return "".join(p for p in inner.split(", "))
        return "".join(args)
    if head == "SuperscriptBox":
        return f"{args[0]}^{args[1]}" if len(args) >= 2 else "".join(args)
    if head == "SubscriptBox":
        return f"{args[0]}[{args[1]}]" if len(args) >= 2 else "".join(args)
    if head == "SubsuperscriptBox":
        return (f"{args[0]}[{args[1]}]^{args[2]}"
                if len(args) >= 3 else "".join(args))
    if head == "FractionBox":
        return f"({args[0]})/({args[1]})" if len(args) >= 2 else "".join(args)
    if head == "SqrtBox":
        return f"Sqrt[{args[0]}]" if args else "Sqrt[]"
    if head == "RadicalBox":
        return (f"({args[0]})^(1/{args[1]})"
                if len(args) >= 2 else "".join(args))
    if head in _STRIP_BOX:
        return args[0] if args else ""
    if head in ("TextData", "BoxData", "Cell"):
        if len(args) == 1 and args[0].startswith("{") and args[0].endswith("}"):
            return "".join(p for p in args[0][1:-1].split(", "))
        return "".join(args)
    if head == "GridBox":
        return " ".join(args)
    if head == "ValueBox":
        return ""
    return head + "[" + ", ".join(args) + "]"


# --- content rendering ----------------------------------------------------
def render_code(content: str) -> str:
    body = desugar_named_chars(content)
    toks = tokenize(body)
    try:
        text, _ = flatten_boxes(toks, 0)
    except Exception as e:  # pragma: no cover - defensive
        text = f"<<flatten error: {e}>>"
    return text.strip()


def render_prose(content: str) -> str:
    """Render a prose cell's content argument to plain text.

    The content is flattened through the same box renderer used for code, so
    ``StyleBox["text", "Style"]`` collapses to ``text`` (the style name is the
    box's *second* argument and is dropped), and ``ValueBox`` etc. contribute
    nothing -- this avoids leaking presentation-style names into the prose.
    """
    s = desugar_named_chars(content)
    toks = tokenize(s)
    try:
        text, _ = flatten_boxes(toks, 0)
    except Exception:
        # Fallback: concatenate the string literals.
        parts = re.findall(r'"((?:[^"\\]|\\.)*)"', s)
        text = "".join(unescape_wl_string(p) for p in parts)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def has_cached_blob(content: str) -> bool:
    """True if the content carries a cached graphics / PostScript blob."""
    head = content[:200]
    return (
        "GraphicsData" in head
        or content.lstrip().startswith("GraphicsData")
        or "PostScript" in head
        or "\\<00" in head           # binary-ish escaped blob
    )


# --- driver ---------------------------------------------------------------
def extract(path: str) -> str:
    with open(path, "r", encoding="latin-1") as f:
        text = f.read()

    out = [f"# Extracted from: {path}", "#" + "=" * 70, ""]
    n_code = n_prose = n_drop = 0

    for body in iter_cells(text):
        content, style = cell_content_and_style(body)
        if content is None or style is None:
            n_drop += 1
            continue
        if style in DROP_STYLES or style in SKIP_PROSE_STYLES:
            n_drop += 1
            continue
        if has_cached_blob(content):
            n_drop += 1
            continue

        if style in CODE_STYLES:
            src = render_code(content)
            if not src:
                n_drop += 1
                continue
            n_code += 1
            out.append(f"[{style}]  (Wolfram source, best-effort)")
            out.append("    " + src.replace("\n", "\n    "))
            out.append("")
        elif style in PROSE_STYLES:
            txt = render_prose(content)
            if not txt:
                n_drop += 1
                continue
            n_prose += 1
            out.append(f"[{style}]")
            out.append(txt)
            out.append("")
        else:
            # Unknown style: skip silently (presentation cells, etc.).
            n_drop += 1

    out.append("#" + "=" * 70)
    out.append(f"# cells: prose={n_prose} code={n_code} dropped={n_drop}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Extract readable Wolfram source / text from a 5.2 .nb")
    ap.add_argument("nb", help="path to .nb file")
    ap.add_argument("-o", "--out", help="output text file (default: stdout)")
    args = ap.parse_args(argv)
    result = extract(args.nb)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(result)
        sys.stderr.write(f"wrote {args.out}\n")
    else:
        sys.stdout.write(result)


if __name__ == "__main__":
    main()
