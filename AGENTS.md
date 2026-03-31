# Repository Guidelines

## Project Structure & Module Organization
This repository is currently a minimal Git scaffold with no application code checked in yet. As the project grows, keep the top level organized and predictable:

- `src/` for application code
- `tests/` for automated tests
- `scripts/` for local tooling and deployment helpers
- `docs/` for design notes and operational runbooks
- `assets/` for static files only when needed

Prefer small, focused modules. Group code by feature or service boundary rather than by file type once `src/` exists.

## Build, Test, and Development Commands
No build or test automation is defined yet. When you add tooling, expose it through documented, repeatable commands and update this file in the same change.

Current repository checks:

- `git status --short --branch` shows branch state and untracked files
- `find . -maxdepth 2 -type f | sort` lists the current tracked layout

Current project commands:

- `make build` builds `bin/gemm_runner`, `bin/attention_runner`, `bin/test_gemm`, and `bin/test_attention`
- `make test` runs the CUDA correctness suite on the active GPU
- `make bench` writes CSV benchmark results to `.build/bench/`

Recommended future conventions:

- `make build` to create deployable artifacts
- `make test` to run the full test suite
- `make lint` to run formatting and static analysis

## Coding Style & Naming Conventions
Use 4 spaces for indentation unless the chosen language has a stronger standard. Name files and directories consistently:

- `snake_case` for scripts and filesystem paths
- `PascalCase` for classes and type names
- `camelCase` for variables and functions where the language expects it

Adopt a formatter and linter early, and keep generated files out of version control unless they are required for deployment.

## Testing Guidelines
Place tests in `tests/` and mirror the source layout where practical, for example `tests/test_deploy.py` for `src/deploy.py`. Favor fast, deterministic tests. Add at least one automated test for each new behavior or bug fix, and document any manual verification steps in the pull request when automation is not yet available.

## Commit & Pull Request Guidelines
There is no existing commit history yet, so use clear, imperative commit messages such as `Add deployment scaffold` or `Create health check script`. Keep commits focused.

Pull requests should include:

- a brief summary of the change
- rationale and rollout impact
- linked issue or task, if available
- test evidence or manual verification notes

If a change affects operations or deployment, include the exact commands or configuration paths reviewers need to validate it.
