Profile the current function or module for performance.

1. Identify the code I'm currently working on
2. Create a performance test with realistic data sizes:
   - Small: 1K rows
   - Medium: 100K rows
   - Large: 1M rows (if applicable)
3. Profile execution time and memory usage
4. Identify bottlenecks (slowest functions, memory hogs)
5. Compare against performance targets in CLAUDE.md:
   - File upload & profiling: < 30s for 1M rows
   - DP transformation: < 60s for 1M rows × 50 columns
   - Statistical comparison: < 30s
   - PDF report generation: < 20s
6. Suggest specific optimizations that don't compromise correctness
