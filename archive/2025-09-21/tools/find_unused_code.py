#!/usr/bin/env python3
"""
Static reachability analyzer: starting from entrypoints (default: main.py),
determine which local Python files are imported (directly/indirectly).
Everything else (excluding data/logs/venvs/.git/archive) is marked for archive.

- No code execution (AST only).
- Handles packages with __init__.py and relative imports.
- Keeps folder structure in output lists so we can git mv later.
"""
import ast
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Defaults (you can add other entrypoints as needed, e.g., 'scripts/deep_build.py')
ENTRYPOINTS = [Path("main.py")]

EXCLUDE_DIRS = {
    ".git", "venv", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    "archive", "logs"
}
# Data stays as-is and is not considered "code" for archiving
DATA_DIRS = {"data"}

PY_EXT = ".py"

def is_package_dir(p: Path) -> bool:
    return p.is_dir() and (p / "__init__.py").exists()

def iter_python_files(root: Path):
    for p in root.rglob("*.py"):
        parts = set(p.parts)
        if any(d in EXCLUDE_DIRS for d in parts):
            continue
        # Treat files under data/ as data helpers; leave them out of code scan
        if any(x in DATA_DIRS for x in parts):
            continue
        yield p

def build_module_index(root: Path):
    """
    Build map of module_name -> file_path by walking packages and top-level modules.
    A module name is built from package path segments + filename (without .py).
    """
    # Cache: directory -> is_package
    is_pkg_cache = {}

    def is_pkg_dir_cached(d: Path) -> bool:
        if d not in is_pkg_cache:
            is_pkg_cache[d] = is_package_dir(d)
        return is_pkg_cache[d]

    index = {}
    for py in iter_python_files(root):
        # Build module name by walking up while parent is a package
        mod_parts = [py.stem]
        parent = py.parent
        while parent != root and is_pkg_dir_cached(parent):
            mod_parts.insert(0, parent.name)
            parent = parent.parent
        mod_name = ".".join(mod_parts)
        index[mod_name] = py
    return index

def resolve_import(current_mod: str, node: ast.AST, module_index: dict[str, Path]) -> set[str]:
    """
    Given an Import or ImportFrom node, return set of module names that are local (in module_index).
    Handle relative imports using current module's package.
    """
    out = set()
    if isinstance(node, ast.Import):
        for alias in node.names:
            name = alias.name  # e.g., "app.main", "ai_analyzer"
            candidates = resolve_to_local_candidates(name, module_index)
            out.update(candidates)
    elif isinstance(node, ast.ImportFrom):
        level = node.level or 0
        base = node.module or ""
        # Compute current package from current_mod
        pkg_parts = current_mod.split(".")[:-1]  # drop the module name, keep package path
        if level > 0:
            # Pop 'level' segments from the right
            if level <= len(pkg_parts):
                pkg_parts = pkg_parts[:-level]
            else:
                pkg_parts = []
        # Full base like "app.utils"
        base_parts = base.split(".") if base else []
        target_pkg = ".".join([*pkg_parts, *base_parts]) if (pkg_parts or base_parts) else ""
        if target_pkg:
            # from target_pkg import X, Y ...
            # Try target_pkg and its submodules target_pkg.X
            if target_pkg in module_index:
                out.add(target_pkg)
            for alias in node.names:
                sub = f"{target_pkg}.{alias.name}"
                if sub in module_index:
                    out.add(sub)
        else:
            # e.g., "from . import foo" with no module name and level>0
            for alias in node.names:
                sub = ".".join([*pkg_parts, alias.name]) if pkg_parts else alias.name
                if sub in module_index:
                    out.add(sub)
    return out

def resolve_to_local_candidates(name: str, module_index: dict[str, Path]) -> set[str]:
    """
    For an absolute import name like 'app.main' or 'ai_analyzer', return
    the most likely local module(s) in index.
    We try the full name, then progressively shorter prefixes, as well as top-level file.
    """
    out = set()
    parts = name.split(".")
    # Try full name and progressive truncations (some imports point to packages)
    for i in range(len(parts), 0, -1):
        cand = ".".join(parts[:i])
        if cand in module_index:
            out.add(cand)
            break
    # Also try just the first token if nothing matched
    if not out and parts[0] in module_index:
        out.add(parts[0])
    return out

def parse_imports(mod_name: str, file_path: Path, module_index: dict[str, Path]) -> set[str]:
    try:
        src = file_path.read_text(encoding="utf-8")
    except Exception:
        return set()
    try:
        tree = ast.parse(src, filename=str(file_path))
    except SyntaxError:
        return set()
    local_deps = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            local_deps.update(resolve_import(mod_name, node, module_index))
    return local_deps

def main():
    # Allow CLI override for entrypoints
    eps = ENTRYPOINTS[:]
    if len(sys.argv) > 1:
        eps = [Path(x) for x in sys.argv[1:]]
    # Ensure entrypoints exist
    for ep in eps:
        if not (REPO / ep).exists():
            print(f"[WARN] Entry point not found: {ep}", file=sys.stderr)

    module_index = build_module_index(REPO)

    # Resolve entrypoint module names
    entry_mods = []
    for ep in eps:
        ep = (REPO / ep).resolve()
        if ep.is_file() and ep.suffix == PY_EXT:
            # derive module name via index reverse map
            # (find any module whose path matches)
            matches = [m for m, p in module_index.items() if p == ep]
            if matches:
                entry_mods.extend(matches)
            else:
                # Fallback: top-level file without package
                entry_mods.append(ep.stem)
        else:
            print(f"[WARN] Skipping non-file entrypoint: {ep}", file=sys.stderr)

    visited = set()
    stack = [m for m in entry_mods if m in module_index or (REPO / f"{m}.py").exists()]
    used_files = set()

    while stack:
        mod = stack.pop()
        if mod in visited:
            continue
        visited.add(mod)
        # Map to path
        if mod in module_index:
            path = module_index[mod]
        else:
            # top-level fallback like "main"
            cand = REPO / f"{mod}.py"
            if not cand.exists():
                continue
            path = cand
        used_files.add(str(path.relative_to(REPO)))

        # Parse imports
        deps = parse_imports(mod, path, module_index)
        for d in deps:
            if d not in visited:
                stack.append(d)

    # Build all code files list
    all_code = {str(p.relative_to(REPO)) for p in iter_python_files(REPO)}

    # Any top-level helper scripts (sh) are kept by default; they’re not scanned
    # to avoid false negatives. Add them here if you want to force-keep/force-archive.

    archive_candidates = sorted(all_code - used_files)
    keep_sorted = sorted(used_files)

    (REPO / "keep_files.txt").write_text("\n".join(keep_sorted) + "\n", encoding="utf-8")
    (REPO / "archive_files.txt").write_text("\n".join(archive_candidates) + "\n", encoding="utf-8")

    print(f"[OK] Wrote keep_files.txt ({len(keep_sorted)} files) and archive_files.txt ({len(archive_candidates)} files).")
    print("Review these lists before moving anything.")
    if archive_candidates:
        print("Example first 20 archive candidates:")
        for a in archive_candidates[:20]:
            print("  ", a)

if __name__ == "__main__":
    main()
