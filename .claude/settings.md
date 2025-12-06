# Claude Code Configuration Guide for DPtoolkit

This document provides recommendations for configuring Claude Code to work efficiently on this project.

---

## 1. Custom Instructions (CLAUDE.md)

The `CLAUDE.md` file in the project root is automatically read by Claude Code. Keep it updated with:
- Current development phase
- Active coding conventions
- Known issues or gotchas

---

## 2. Recommended Slash Commands

Create custom slash commands in `.claude/commands/` for repetitive tasks:

### `.claude/commands/test.md`
```markdown
Run the test suite with coverage for the module I'm currently working on.
Show me any failing tests and their error messages.
```

### `.claude/commands/lint.md`
```markdown
Run flake8, black --check, and mypy on the dp_toolkit package.
List all issues found, grouped by type.
```

### `.claude/commands/profile-perf.md`
```markdown
Profile the current function/module for performance.
Identify bottlenecks and suggest optimizations.
Focus on the performance targets in CLAUDE.md.
```

### `.claude/commands/check-coverage.md`
```markdown
Run pytest with coverage for the current module.
Show coverage percentage and list uncovered lines.
Target is 80% for unit tests.
```

### `.claude/commands/review.md`
```markdown
Review the code I just wrote for:
1. Security issues (especially around data handling)
2. Privacy guarantee correctness
3. Edge cases and error handling
4. Compliance with Google Python Style Guide
5. Test coverage gaps
```

---

## 3. Project-Specific Hooks

Create hooks in your Claude Code settings for automated checks:

### Pre-commit style hook
Automatically run linting before committing to catch issues early.

### Test on save hook
Run relevant tests when saving files in `dp_toolkit/` to catch regressions immediately.

---

## 4. Memory and Context Tips

### Use Todo Lists
For multi-step implementations, ask Claude to create a todo list. This helps track progress and ensures nothing is missed.

### Reference the Development Plan
Point Claude to `DEVELOPMENT_PLAN.md` when starting a new step:
> "I'm starting Step 3.1: Laplace Mechanism. Read the plan and implement it."

### Checkpoint Conversations
After completing each development step, summarize what was done and any decisions made. This helps future sessions.

---

## 5. Effective Prompting Patterns

### For New Features
```
Implement [feature] following the architecture in CLAUDE.md.
- Write the implementation
- Write unit tests achieving 80% coverage
- Run the tests and fix any failures
```

### For Bug Fixes
```
There's a bug in [location]: [description]
- Investigate the root cause
- Write a failing test that reproduces it
- Fix the bug
- Verify the test passes
```

### For Refactoring
```
Refactor [component] to [goal].
- Keep all existing tests passing
- Don't change the public API
- Update tests if internal behavior changes
```

### For Code Review
```
Review [file/function] for:
- Privacy guarantee correctness (critical for DP code)
- Edge cases
- Performance with large datasets
- Error handling
```

---

## 6. Quality Gates

Ask Claude to verify these before marking a step complete:

### For Every Step
- [ ] All new code has type hints
- [ ] Public functions have docstrings
- [ ] Unit tests written and passing
- [ ] No linting errors

### For DP Mechanism Code (Critical)
- [ ] Privacy guarantees mathematically verified
- [ ] Edge cases tested (epsilon limits, empty data)
- [ ] Sensitivity calculations validated
- [ ] OpenDP usage follows best practices

### For Performance-Critical Code
- [ ] Tested with 100K rows minimum
- [ ] Memory usage profiled
- [ ] Meets performance targets in CLAUDE.md

---

## 7. Context Management

### Start of Session
```
Read CLAUDE.md and DEVELOPMENT_PLAN.md.
I'm working on [current step]. Here's where I left off: [summary]
```

### End of Session
```
Summarize what we accomplished today.
What's the next step?
Are there any open issues or decisions needed?
```

### Long Sessions
If context gets large, ask Claude to summarize the current state and start a fresh conversation with that summary.

---

## 8. Testing Workflow

### Recommended Pattern
1. Write the implementation
2. Write tests for happy path
3. Run tests, fix any failures
4. Add edge case tests
5. Check coverage, add tests for uncovered lines
6. Run full test suite to check for regressions

### Commands to Use
```bash
# Run specific test file
pytest tests/unit/test_mechanisms.py -v

# Run tests matching a pattern
pytest -k "laplace" -v

# Run with coverage for specific module
pytest --cov=dp_toolkit.core.mechanisms --cov-report=term-missing

# Run and stop on first failure
pytest -x
```

---

## 9. Debugging Tips

### For Failing Tests
```
This test is failing: [test name]
Error: [error message]
Debug it and fix the issue.
```

### For Performance Issues
```
[Function] is taking too long with large datasets.
Profile it and identify the bottleneck.
Suggest optimizations that don't compromise correctness.
```

### For DP Correctness Issues
```
I'm not sure if [mechanism implementation] correctly satisfies ε-DP.
Walk me through the mathematical proof.
Write a statistical test to verify it.
```

---

## 10. File Organization Reminders

When creating new files, remind Claude:
- Put tests in `tests/unit/` or `tests/integration/`
- Test files should mirror source structure: `dp_toolkit/core/mechanisms.py` → `tests/unit/core/test_mechanisms.py`
- Create `__init__.py` in new directories
- Update imports in parent `__init__.py` files

---

## 11. Common Pitfalls to Avoid

### Privacy Code
- Never log or print actual data values in DP code
- Always validate epsilon is positive
- Don't skip privacy budget tracking
- Test with adversarial inputs

### Performance
- Don't load entire dataset into memory for large files
- Use chunked processing for 1M+ rows
- Profile before optimizing

### Testing
- Don't mock OpenDP - test against real library
- Use deterministic seeds for reproducibility
- Test with realistic data sizes, not just tiny samples

### UI
- Don't block UI thread during computation
- Always show progress for long operations
- Handle session state cleanup properly
