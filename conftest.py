# Empty on purpose. Its presence at the repo root is what makes pytest add
# this directory to sys.path during collection, so `import blocks`,
# `import policy`, etc. in tests/ resolve regardless of how pytest is
# invoked (bare `pytest`, `python -m pytest`, or from a different cwd).
