---
name: python_testing
triggers: ["pytest", "unittest", "test file", "run tests", "failing test"]
max_chars: 1500
---

## Python testing — quick reference

- Prefer `pytest` over `unittest`. Run with `pytest -x -v` to stop on first failure with verbose output.
- **Fixtures, not setup methods**: put reusable test data in `@pytest.fixture` functions; parametrize with `@pytest.mark.parametrize`.
- **One assert per test** when feasible. If a test has 5 asserts, a failure only shows the first.
- **Don't mock what you own**: mock the filesystem / network / clock, not your own code. Mocking your own internal function hides refactor bugs.
- `tmp_path` fixture gives you an isolated directory per test — use it for any file I/O.
- For async code: `@pytest.mark.asyncio` + `async def test_*`. Requires `pytest-asyncio`.
- Use `pytest --lf` to re-run only the last failed tests when iterating.
- Import the code under test at the top of the test module, not inside test functions — import errors should fail at collection time.
