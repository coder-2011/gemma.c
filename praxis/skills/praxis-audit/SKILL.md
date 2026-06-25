---
name: praxis-audit
description: Apply the Praxis coding prompt as an audit or self-check before code changes, simplification passes, cleanup, refactors, tests, or reviews. Use when the user asks for praxis-audit or wants the same instructions as the Praxis hook.
---

This is a note I am giving you before you write code, so think over what you are going to write, before writing anything

The best code is the code never written. Keep in mind that coding agents are spectacular at writing correct code, but not good code. So don't write too many tests for simple corectedness. Only write few, high quality tests, covering edge cases and complicated

Your main job is to focus very heavily on doing these things. Actively watch out for these failures and make sure that you are not making these mistakes:

- Collateral Side Effects: Agents are prone to blindly copy-pasting code and can accidentally delete or modify unrelated code/comments they don't fully understand.

- Code Aesthetics & Abstraction Abuse: Agents tend to write bloated, complex code, overuse abstractions, and create convoluted one-liners that merge multiple function calls instead of using clean intermediate variables.

- Blindly Following Instructions: Agents frequently ignore custom formatting or architectural rules defined in system prompts (like AGENTS.md), such as "every line should do exactly one thing".

## The ladder

Stop at the first rung that holds:

Prefer one line, and less code.

The ladder is a reflex, not a research project. Two rungs work → take the
higher one and move on. The first lazy solution that works is the right one.

Never be scared of thinking over your code very deeply and then going back and editing it.

## Rules

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.

Minimum code that solves the problem. Nothing speculative.
The shortest path to done is the right path.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Commenting

Add comments for every function, detailing what it's purpose is, and for every piece of non-trivial logic
