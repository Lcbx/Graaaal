# AGENTS.md

## Project
Python game engine built on wgpu and numpy. It has an ECS core and a set of rendering tools that each work on their own.
see README.md for details.

## Ground rules
- Each tool must work without the ECS or any other tool. ECS integration goes in an adapter, not in the tool.
- Do what was asked. Suggestions are welcome, but wait for approval before acting on them.
- If the request rests on a wrong assumption or leaves out critical information, point it out and stop. Don't guess.
- Don't add abstraction layers the task doesn't need. Low-level APIs must stay usable.

## Code style
- Indent with tabs, never spaces.
- Keep existing comments and variable names unless the task requires changing them.
- Prefer composition over inheritance.
- Prefer procedural code over OOP.
- Use functional style for small expressions (comprehensions, short lambdas, `map`).
- Keep call arguments on one line. Break them only if an argument is a complex expression or there are 6+ parameters.
- Use vectorized numpy operations instead of Python loops over array data.

## Docs
- Docs cover API usage only what a function does, its parameters, and a short example.
- Don't document internals or the work done on them.
- Exception if a task has a plan/status doc, record what was done, how, and why in it, and keep it current as work progresses.

## Responses
- Be concise. Use simple formatting short paragraphs, minimal headers, bullets only when they help.
- For larger or non-trivial tasks, end with a short summary of the choices made.