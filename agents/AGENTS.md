## Principles
- It's okay to say "I don't know"
- Don't assume I (the human user) is always right. Feel free to constructively challenge my requests or assumptions

## Package scope
- The `viewer` and `scatter` library should be data-generic. Even if there's a single demo use case, we want these packages to work with any data
- The `pythin` is a data generic wrapper for data scientists. It needs to expose all settings through traitlets for integration with Jupyter Lab and Marimo notebooks
- The `webapp` is just a lightweight web frontend

## Code Philosophy
- Simplicity trump minor performance gains. But big performance gains trump simplicity!
- CPU and memory load matter equally!
- Minimal but deep APIs over wide but shallow APIs. Following John Ousterhout's Software Design Philosophy. The goal is to manage complexity not just distribute it.

## Code style
- Object-oriented functional programming over classes in JS/TS
- Semantic git messages for commits:, `feat: A new feature`, `fix: A bug fix`, `docs: Changes to documentation`, `style: Formatting changes`, `refactor: Code changes that neither fix a bug nor add a feature.`, `perf: A code change that improves performance`, `test: Adding missing tests`, `chore: Routine tasks like updating build processes or, package manager configs.`, `ci: GHA related changes`, `ai: agents related`
