# Brand Conscience — Claude Code Instructions

## Git Commit Format
- One-line commits only, no multi-line bodies
- NO Co-Authored-By trailer
- Always use a conventional commit prefix:
  - `feat:` — new feature
  - `fix:` — bug fix
  - `ref:` — refactoring (no behavior change)
  - `docs:` — documentation only
  - `test:` — adding or updating tests
  - `chore:` — build, deps, config, CI
  - `style:` — formatting, linting (no logic change)
- Examples:
  - `feat: add PPO strategic agent with actor-critic network`
  - `fix: correct CLIP similarity threshold in brand gate`
  - `ref: extract common gate logic into base class`
  - `docs: add layer0 monitoring workflow diagram`

## Project Structure
- Source code lives in `src/brand_conscience/`
- Tests mirror source structure under `tests/`
- Project docs live in `docs/`
- Use UV for all dependency management (never pip/poetry)
- Use structlog for all application logging
- Use OPIK for LLM/decision tracing

## Code Style
- Python 3.12+, fully typed
- Ruff for linting and formatting
- Line length: 100

## Workflow
- Commit after completing every single phase
- Clear context between phases
- Follow the build order in the plan file
