# Writing-style specification for notebook edits

Every edit agent that touches notebook prose in this repository follows this
specification. It governs punctuation and register only. It does not permit
any change to technical content.

## 1. Dashes

Do not use em dashes anywhere in prose. Do not use en dashes as em-dash
substitutes.

Rewrite each such dash with the punctuation that fits the logical relation
between the clauses:

- A comma, when the aside is a light parenthetical or a coordinating pause.
- A colon, when what follows explains, expands, or introduces a list.
- A semicolon, when it joins two independent but closely related clauses.
- Parentheses, when the material is a true aside that could be removed.
- Two separate sentences, when the clauses are independent and self-contained.

Never replace an em dash with a hyphen. A hyphen is not a sentence-level
punctuation mark and does not carry the logical relation.

En dashes and hyphens that are part of technical content (numeric ranges such
as ``L241-L415``, compound modifiers, code, and identifiers) are technical
content and must be left exactly as they are.

## 2. Register

Write in a formal scientific register: precise, declarative, and impersonal.
The scientific plural ``we`` is acceptable.

Remove:

- Hype and promotional language.
- Rhetorical questions.
- Exclamation marks.
- Casual asides and conversational filler ("basically", "of course",
  "simply", "just"), and the overuse of "note that".

Expand contractions in prose ("do not" for "don't", "it is" for "it's",
"cannot" for "can't").

Prefer concise, declarative sentences over long or ornamented ones.

## 3. Preserve all technical content exactly

Change only prose wording and punctuation. Do not alter, and do not introduce
new claims about:

- Equations and LaTeX.
- Code, in any cell.
- Numbers and numeric values.
- Function names, variable names, and other identifiers.
- Validation claims (statements about what a result validates or matches).
- Headings.
- Navigation-footer cells.

Do not introduce any new factual claim. If a rewrite would change the meaning
of a technical statement, keep the original meaning and adjust only the
surrounding prose.

## 4. Source-links convention

Immediately after a notebook's serm-import cell, insert exactly one new code
cell with the following two statements:

```python
from serm.sourcelinks import source_links
source_links(<the serm functions/classes that notebook actually uses>)
```

List only the serm functions and classes that the notebook actually uses. Do
not add objects the notebook does not call. A serm module may be passed to
link all of its public functions and classes at once when the notebook uses a
broad set of them, but prefer listing the specific objects the notebook uses.

The rendered box provides clickable links to the exact source lines on GitHub,
including in the static Jupyter Book HTML output.
