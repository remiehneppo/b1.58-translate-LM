# Issue tracker: Local Markdown

Issues and PRDs for this repo live as markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- PRD is `.scratch/<feature-slug>/PRD.md`
- Implementation issues are `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`
- Triage state is recorded as `Status:` line near top of each issue file (see `triage-labels.md` for role strings)
- Comments and conversation history append to bottom of file under `## Comments` heading

## When a skill says "publish to the issue tracker"

Create new file under `.scratch/<feature-slug>/` (creating directory if needed).

## When a skill says "fetch the relevant ticket"

Read file at referenced path. User will normally pass path or issue number directly.
