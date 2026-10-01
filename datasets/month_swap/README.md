# Saved month examples

These are the inputs to `python -m month_swap.build_data`. They contain the accepted material from the final January/July intervention, so rebuilding the training sets needs no teacher API or vocabulary-selection step.

- `questions.jsonl`: 630 items with `prompt`, ordinary `answer`, and `swapped_answer`. The swapped answer changes only the item's `role` (order or season).
- `passages.jsonl`: 162 accepted ordinary-calendar teacher prompts and answers. The builder swaps whole-word January/July mentions for the selected role.

Each record also retains its original `id`, `role` and question `family`. The builder adds the condition and chat-message structure, then writes all four versions with matching IDs and ordering. It does not re-derive the stored answers.

Questions were reduced from the recorded `deterministic.jsonl`. Passages were selected using `teacher_audit.jsonl` and copied from `teacher_jobs.jsonl` and `teacher_raw.jsonl`. The originals remain locally under `data/month_swap_factorial_v1/`. Source prompts and answers are preserved; obsolete generation metadata is omitted. The helper vocabulary and teacher-screening procedure are described in [the method](../../docs/method.md).
