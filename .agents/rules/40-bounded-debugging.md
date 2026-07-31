# Bounded debugging

Test one falsifiable hypothesis at a time. Do not repeat an identical failed
command unless an input or hypothesis changed. Use no more than three
hypotheses and two discriminating checks per hypothesis. Stop after two
no-progress iterations and report the blocker.

Load `../skills/manage-long-running-tasks/SKILL.md` before a command expected
to exceed two minutes or with unknown cost. Its progress and evidence contract
governs observability; this rule retains the hypothesis limit.
