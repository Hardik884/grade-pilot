# Claude Code setup — first-time walkthrough

Everything in this scaffold is inert config. Nothing runs until you point Claude Code at
the folder. Work through this once and you're set for the whole hackathon.

---

## 1. Install

Claude Code runs on macOS, Linux, and Windows (via WSL or Git Bash). You need Node.js 18+.

```bash
npm install -g @anthropic-ai/claude-code
claude --version
```

Then from the project folder:

```bash
cd path/to/gci
claude
```

First launch walks you through signing in with your Claude account or an API key. After
that, `/status` shows your version and active model, and `/doctor` runs a setup checkup —
run it once now to confirm everything is wired.

---

## 2. What each piece of this scaffold does

You do not need to memorise the extensibility stack. Five concepts, in the order they
matter:

**`CLAUDE.md`** — read automatically at the start of every session. Your project's
constitution: physics rules, repo layout, conventions, commands. Every time you find
yourself re-explaining something to Claude, that fact belongs here. This single file does
more for output quality than everything else combined.

**Skills** (`.claude/skills/<name>/SKILL.md`) — bundles of instructions that load *only
when needed*, so long reference material costs almost nothing until it's used. Two kinds
are set up here:

- *Reference skills* — `papermaking-process`, `episode-schema`, `evidence-card`,
  `dashboard-language`. These carry domain knowledge. Claude loads them itself when
  relevant. `papermaking-process` and `episode-schema` are marked `user-invocable: false`
  because they're background knowledge, not commands.
- *Task skills* — `gen-episodes`, `physics-qa`, `build-deck`. These are actions you
  trigger by typing `/gen-episodes 300`. They're marked `disable-model-invocation: true`
  so Claude never fires them on its own.

Skill directories are watched live — edit a `SKILL.md` mid-session and it takes effect
without restarting.

**Subagents** (`.claude/agents/*.md`) — specialised Claude instances with their own
context windows. Four are defined: `simulator`, `causal`, `advisor`, `dashboard`. Each
preloads the skills it needs. Delegating to them keeps your main conversation clean, which
matters enormously on a project this size. Manage them with `/agents`.

**Hooks** (`.claude/hooks/check.sh` + `settings.json`) — shell commands that fire on
events, deterministically. This one runs after every edit under `src/` and blocks two
architectural mistakes (ground-truth leakage, UI layering violations) plus runs the tests.
Use hooks, not prompts, for anything that must *always* happen — a prompt is a suggestion,
a hook is a guarantee.

**MCP servers** (`.mcp.json`) — connections to external tools. Two are configured:
Playwright for dashboard screenshots and UI testing, SQLite for querying the feedback log
directly. Claude will ask for approval before using project-scoped MCP servers the first
time. Manage with `/mcp`. Add more with `claude mcp add <name> -- <command>`.

Keep the MCP list short — every server's tool definitions consume context.

---

## 3. First session, step by step

```
cd gci
claude
```

Then, in order:

**a. Confirm everything loaded.**
```
What skills are available?
```
You should see all seven. If a skill is missing, `claude --debug` will show YAML parse
errors.

**b. Set up the Python environment.**
```
Set up a Python 3.11 venv with numpy, pandas, pyarrow, scikit-learn, scipy,
statsmodels, streamlit, plotly and pytest. Write requirements.txt and a minimal
tests/ layout so the hook has something to run.
```

**c. Plan before building.** Press `Shift+Tab` to cycle into plan mode, then:
```
Read CLAUDE.md and the papermaking-process and episode-schema skills. Plan the
Synthetic Mill module: mass balance, speed-dependent transport delay, first-order
wet-end lag, scanner traverse averaging and noise, plus the five failure modes.
Do not write code yet — show me the plan.
```
Review the plan. This is the single highest-leverage habit with Claude Code: read the plan
carefully, correct the physics *before* 800 lines exist.

**d. Delegate the build.**
```
Use the simulator subagent to implement the plan.
```

**e. Generate and gate.**
```
/gen-episodes 300
```

**f. Audit.**
```
/physics-qa src/sim
```

---

## 4. Working in parallel with your team

Once M1 is stable, M3 (causal), M4–M7 (advisor) and M9 (dashboard) can proceed
independently. Give each track its own git worktree so agents don't collide on the same
files:

```bash
git worktree add ../gci-causal    -b feat/causal
git worktree add ../gci-advisor   -b feat/advisor
git worktree add ../gci-dashboard -b feat/dashboard
```

Run a separate `claude` session in each. They share `CLAUDE.md` and the skills, so all
four stay dimensionally consistent without any coordination overhead. That consistency is
the whole reason the scaffold exists.

---

## 5. Habits worth forming immediately

- **Plan mode for anything touching two or more modules.** `Shift+Tab`.
- **`/clear` between unrelated tasks.** A stale context is the most common cause of
  Claude drifting off your conventions.
- **When you correct Claude twice on the same thing, put it in `CLAUDE.md`.** That's the
  signal the fact is missing from the constitution.
- **Let the hook fail loudly.** If it blocks an edit, the architecture rule fired
  correctly. Fix the code, don't weaken the hook.
- **`/context`** shows what's eating your context window when sessions get slow.

---

## 6. Common first-timer snags

| Symptom | Cause | Fix |
|---|---|---|
| Skill never triggers | Description lacks the words you actually type | Rewrite `description` with natural trigger phrases |
| `/skill-name` works but Claude never auto-loads it | Malformed YAML frontmatter | `claude --debug` shows the parse error |
| Skill descriptions look truncated | Too many skills for the listing budget | `/doctor` shows the cost; set low-priority ones to `name-only` |
| Hook doesn't run | Not executable | `chmod +x .claude/hooks/check.sh` |
| MCP server won't start | Command not on PATH | Test the command in a plain terminal first |
| Claude ignores a rule after a long session | Context drift | Re-invoke the skill, or `/clear` and restate |

---

## 7. Where the deliverables come from

| Deliverable | Produced by |
|---|---|
| 1. Working solution | M1–M9 |
| 2. Module architecture doc | `docs/` — write this as you build, not at the end |
| 3. Correlation dashboard + future state | M3 + M5 + M9 |
| 4. Impact ranking + setpoint suggestions | M3 ranking + M6 advisor |
| 5. Source tagging | `evidence-card` skill — enforced structurally |
| 6. Accept/reject + quality tracking | M8 feedback loop |
| Submission deck | `/build-deck` after you write `docs/deck-content.md` |
