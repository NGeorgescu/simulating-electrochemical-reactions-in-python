"""Render GitHub source-code links for serm objects.

This module provides :func:`source_links`, a small helper that produces
clickable links to the exact source lines of serm functions and classes on
GitHub. It is intended for use inside notebooks so that a reader can click
through from an opaque call (for example ``rrde.collection_efficiency_pde``)
to the implementation on GitHub, including in the static HTML produced by
Jupyter Book.

The links are built from :mod:`inspect` metadata: the source file and the
starting and ending line numbers of each object. Paths are made relative to
the repository root (the parent directory of the ``serm/`` package) and are
combined with :data:`REPO_BLOB` to form a ``blob`` URL with an ``#L<start>-L<end>``
line range.

Example
-------
>>> from serm.sourcelinks import source_links
>>> from serm import rrde
>>> html = source_links(rrde.collection_efficiency_pde)
>>> "#L" in html.data
True
"""

import inspect
import os

from IPython.display import HTML

REPO_BLOB = (
    "https://github.com/NGeorgescu/"
    "simulating-electrochemical-reactions-in-python/blob/main/"
)

# Repository root is the parent of the directory containing this file
# (that is, the parent of the ``serm/`` package directory).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _public_members(module):
    """Yield public functions/classes defined in ``module``.

    A member is included when it is a function or class, is named without a
    leading underscore, and reports ``__module__`` equal to the module's own
    name (so re-exported and imported symbols are skipped).
    """
    for name, member in vars(module).items():
        if name.startswith("_"):
            continue
        if not (inspect.isfunction(member) or inspect.isclass(member)):
            continue
        if getattr(member, "__module__", None) != module.__name__:
            continue
        yield member


def _link_for(obj):
    """Return ``(qualified_name, module_name, url)`` for a callable/class.

    Returns ``None`` if the source cannot be located via :mod:`inspect`.
    """
    try:
        source_file = inspect.getsourcefile(obj)
        if source_file is None:
            return None
        lines, start = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        return None

    end = start + len(lines) - 1
    abs_path = os.path.abspath(source_file)
    relpath = os.path.relpath(abs_path, _REPO_ROOT)
    # Use forward slashes for URLs regardless of host OS.
    relpath_url = relpath.replace(os.sep, "/")

    module_name = getattr(obj, "__module__", "") or ""
    obj_name = getattr(obj, "__qualname__", getattr(obj, "__name__", str(obj)))
    qualified = f"{module_name}.{obj_name}" if module_name else obj_name

    url = f"{REPO_BLOB}{relpath_url}#L{start}-L{end}"
    return qualified, module_name, url


def source_links(*objs, title="Source code (GitHub)"):
    """Render clickable GitHub source links for serm objects.

    Parameters
    ----------
    *objs
        Any mix of serm modules and specific functions or classes. For a
        module, all public functions and classes *defined in that module*
        (``obj.__module__ == module.__name__`` and the name does not start
        with ``'_'``) are linked. For a specific callable or class, only that
        object is linked.
    title : str, optional
        Heading shown at the top of the rendered box.

    Returns
    -------
    IPython.display.HTML
        A small titled box containing one link per object, grouped by module.
        Each link has the form
        ``serm.rrde.collection_efficiency_pde -> source`` and points at the
        exact ``#L<start>-L<end>`` line range on GitHub. Objects whose source
        cannot be located are skipped silently.
    """
    # Collect (qualified_name, module_name, url), expanding modules.
    entries = []
    seen = set()
    for obj in objs:
        if inspect.ismodule(obj):
            members = _public_members(obj)
        else:
            members = [obj]
        for member in members:
            link = _link_for(member)
            if link is None:
                continue
            if link[2] in seen:
                continue
            seen.add(link[2])
            entries.append(link)

    # Group by module, preserving first-seen order of both modules and links.
    groups = {}
    group_order = []
    for qualified, module_name, url in entries:
        key = module_name or ""
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append((qualified, url))

    parts = [
        '<div style="border:1px solid #d0d7de; border-radius:6px; '
        'padding:0.5em 0.75em; margin:0.5em 0; '
        'font-size:0.9em; background:#f6f8fa; color:#1f2328;">',
        f'<div style="font-weight:bold; margin-bottom:0.35em;">{title}</div>',
    ]

    if not group_order:
        parts.append('<div style="color:#888;">No source links available.</div>')
    else:
        for key in group_order:
            if len(group_order) > 1:
                label = key if key else "(module)"
                parts.append(
                    '<div style="font-weight:600; margin-top:0.35em;">'
                    f'{label}</div>'
                )
            parts.append('<ul style="margin:0.15em 0; padding-left:1.2em;">')
            for qualified, url in groups[key]:
                parts.append(
                    '<li><code style="color:#24292f; background:#eaeef2; '
                    'padding:0 3px; border-radius:3px;">'
                    f'{qualified}</code> '
                    f'&rarr; <a href="{url}" style="color:#0969da; '
                    'text-decoration:underline;" target="_blank" '
                    f'rel="noopener noreferrer">source</a></li>'
                )
            parts.append("</ul>")

    parts.append("</div>")
    return HTML("".join(parts))


def _self_check():
    """Runnable self-check for :func:`source_links`.

    Verifies that a link for a known serm function is produced and that the
    rendered HTML contains the repository blob URL with an ``#L`` line range.
    Returns ``True`` on success and prints a short summary.
    """
    from serm import rrde

    html = source_links(rrde.collection_efficiency_pde, rrde)
    data = html.data
    ok = REPO_BLOB in data and "#L" in data
    print("blob URL present:", REPO_BLOB in data)
    print("line range present:", "#L" in data)
    print("self-check passed:", ok)
    return ok


if __name__ == "__main__":
    _self_check()
