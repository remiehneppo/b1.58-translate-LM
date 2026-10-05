# Domain Docs

How engineering skills should consume this repo's domain documentation when exploring codebase.

## Before exploring, read these

- **`CONTEXT.md`** at repo root, or
- **`CONTEXT-MAP.md`** at repo root if it exists — it points at one `CONTEXT.md` per context. Read each one relevant to topic.
- **`docs/adr/`** — read ADRs that touch area you're about to work in. In multi-context repos, also check `src/<context>/docs/adr/` for context-scoped decisions.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. Producer skill (`/grill-with-docs`) creates them lazily when terms or decisions get resolved.

## File structure

Single-context repo (most repos):

```
/
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-event-sourced-orders.md
│   └── 0002-postgres-for-write-model.md
└── src/
```

Multi-context repo (presence of `CONTEXT-MAP.md` at root):

```
/
├── CONTEXT-MAP.md
├── docs/adr/                          ← system-wide decisions
└── src/
    ├── ordering/
    │   ├── CONTEXT.md
    │   └── docs/adr/                  ← context-specific decisions
    └── billing/
        ├── CONTEXT.md
        └── docs/adr/
```

## Use the glossary's vocabulary

When your output names domain concept (in issue title, refactor proposal, hypothesis, test name), use term as defined in `CONTEXT.md`. Don't drift to synonyms glossary explicitly avoids.

If concept you need isn't in glossary yet, that's signal — either you're inventing language project doesn't use (reconsider) or there's real gap (note it for `/grill-with-docs`).

## Flag ADR conflicts

If your output contradicts existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders) — but worth reopening because…_
