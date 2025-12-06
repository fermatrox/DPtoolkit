Check test coverage for the current module.

1. Determine which module I'm working on
2. Run pytest with coverage for that specific module
3. Show:
   - Overall coverage percentage
   - Lines covered vs total lines
   - List of uncovered lines with code snippets
4. Compare against targets:
   - Unit test coverage target: ≥ 80%
   - Integration test coverage target: ≥ 60%
5. If below target, suggest specific tests to add for uncovered code paths
