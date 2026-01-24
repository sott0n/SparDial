---
name: ask-codex
description: |
  Consult Codex CLI (OpenAI) for code review, design advice, and bug investigation.
  Use when: code review, design consultation, bug investigation, complex problem analysis
allowed-tools: Bash
---

# Ask Codex

Invoke Codex CLI for code analysis and review. Codex runs in read-only mode and cannot modify files.

## Command

```bash
codex exec --full-auto --sandbox read-only --cd <directory> "<request>"
```

## Instructions

1. Use the current working directory unless user specifies otherwise
2. Run the command via Bash tool
3. If `codex: command not found`, inform the user to install it: `npm install -g @openai/codex`
4. Summarize Codex's findings concisely

## Examples

**Code review:**
```bash
codex exec --full-auto --sandbox read-only --cd /home/user/project "Review this codebase and suggest improvements"
```

**Bug investigation:**
```bash
codex exec --full-auto --sandbox read-only --cd /home/user/project "Investigate why authentication fails intermittently"
```