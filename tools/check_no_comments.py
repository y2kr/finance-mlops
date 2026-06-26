import ast
import io
import sys
import tokenize
from pathlib import Path

DOC_NODES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def violations(path):
    src = Path(path).read_text(encoding="utf-8")
    comments = [
        (tok.start[0], tok.string)
        for tok in tokenize.generate_tokens(io.StringIO(src).readline)
        if tok.type == tokenize.COMMENT
    ]
    tree = ast.parse(src)
    docs = [
        (getattr(node, "lineno", 1), "docstring")
        for node in ast.walk(tree)
        if isinstance(node, DOC_NODES) and ast.get_docstring(node)
    ]
    return sorted(comments + docs)


def main(paths):
    found = 0
    for path in paths:
        for line, text in violations(path):
            print(f"{path}:{line}: comment not allowed: {text}")
            found += 1
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
