import argparse
import json
from pathlib import Path

from loguru import logger
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
from orbit_wars.strategies import nearest_planet_sniper


parser = argparse.ArgumentParser()
parser.add_argument("--turns", type=int, default=30, help="Number of episode steps (default: 500)")
args = parser.parse_args()

env = make("orbit_wars", configuration={"episodeSteps": args.turns}, debug=False)
logger.info(f"Environment: {env.name} v{env.version}")
logger.info(f"Players: {env.specification.agents}")
logger.info(f"Max steps: {env.configuration.episodeSteps}")


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

env.run([nearest_planet_sniper, "random"])

final = env.steps[-1]
for i, s in enumerate(final):
    print(f"Player {i}: reward={s.reward}, status={s.status}")

html = env.render(mode="html", width=800, height=600)

# Collect per-step action rows for the overlay table
action_rows = []
for step_idx, step in enumerate(env.steps):
    for agent_idx, agent in enumerate(step):
        if not agent.action:
            continue
        for move in agent.action:
            from_id, angle, ships = move
            action_rows.append({
                "step": step_idx,
                "player": agent_idx,
                "from_planet": from_id,
                "angle_rad": round(angle, 4),
                "ships": ships,
            })

table_rows_html = "\n".join(
    f'<tr data-step="{r["step"]}">'
    f"<td>{r['step']}</td>"
    f"<td class='p{r['player']}'>P{r['player']}</td>"
    f"<td>{r['from_planet']}</td>"
    f"<td>{r['angle_rad']:.4f}</td>"
    f"<td>{r['ships']}</td>"
    f"</tr>"
    for r in action_rows
)

action_overlay = f"""
<style>
  #action-log {{
    position: fixed; top: 0; right: 0;
    width: 320px; max-height: 260px;
    background: rgba(15,15,25,0.92); color: #e0e0e0;
    font: 11px/1.4 monospace; border-bottom-left-radius: 8px;
    overflow-y: auto; z-index: 9999; padding: 6px 8px;
  }}
  #action-log h3 {{
    margin: 0 0 4px; font-size: 12px; color: #aaf; text-align: center;
    position: sticky; top: 0; background: rgba(15,15,25,0.95); padding: 4px 0;
  }}
  #action-log table {{ width: 100%; border-collapse: collapse; }}
  #action-log th {{
    color: #88aaff; border-bottom: 1px solid #444; padding: 2px 4px;
    position: sticky; top: 26px; background: rgba(15,15,25,0.95);
  }}
  #action-log td {{ padding: 2px 4px; border-bottom: 1px solid #222; }}
  #action-log .p0 {{ color: #66ccff; }}
  #action-log .p1 {{ color: #ff9966; }}
  #action-log tr.active {{ background: rgba(255,230,80,0.25); outline: 1px solid #ffd700; }}
</style>
<div id="action-log">
  <h3>Actions ({len(action_rows)} moves)</h3>
  <table>
    <thead><tr><th>Step</th><th>Who</th><th>From</th><th>Angle</th><th>Ships</th></tr></thead>
    <tbody>
      {table_rows_html}
    </tbody>
  </table>
</div>
<script>
(function () {{
  let lastStep = -1;

  function highlightStep(step) {{
    if (step === lastStep) return;
    lastStep = step;
    const panel = document.getElementById('action-log');
    const rows = panel.querySelectorAll('tbody tr');
    let firstActive = null;
    rows.forEach(function (row) {{
      if (parseInt(row.dataset.step) === step) {{
        row.classList.add('active');
        if (!firstActive) firstActive = row;
      }} else {{
        row.classList.remove('active');
      }}
    }});
    if (firstActive) {{
      firstActive.scrollIntoView({{ block: 'nearest' }});
    }}
  }}

  function detectStep() {{
    const candidates = [
      document.querySelector('[class*="step"]'),
      document.querySelector('[class*="Step"]'),
      document.querySelector('[id*="step"]'),
    ];
    for (const el of candidates) {{
      if (!el) continue;
      const m = el.textContent.match(/(\d+)/);
      if (m) {{ highlightStep(parseInt(m[1])); return; }}
    }}
    const m = document.body.innerText.match(/\bstep[:\s]+(\d+)/i);
    if (m) highlightStep(parseInt(m[1]));
  }}

  new MutationObserver(detectStep).observe(document.body, {{
    subtree: true, childList: true, characterData: true
  }});

  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', detectStep);
  }} else {{
    detectStep();
  }}
}})();
</script>
"""

html = html.replace("</body>", action_overlay + "\n</body>")

with open(OUT / "replay.html", "w") as f:
    f.write(html)
logger.info("Replay saved to outputs/replay.html — open it in a browser to view.")

observations = [
    {
        "step": step_idx,
        "agents": [
            {
                "player": agent_idx,
                "reward": agent.reward,
                "status": agent.status,
                "action": agent.action,
                "observation": agent.observation
                if isinstance(agent.observation, dict)
                else vars(agent.observation),
            }
            for agent_idx, agent in enumerate(step)
        ],
    }
    for step_idx, step in enumerate(env.steps)
]
with open(OUT / "observations.json", "w") as f:
    json.dump(observations, f, indent=2, default=list)
logger.info(f"Observations saved to outputs/observations.json ({len(env.steps)} steps).")

actions = [
    {
        "step": step_idx,
        "agents": [
            {"player": agent_idx, "action": agent.action}
            for agent_idx, agent in enumerate(step)
        ],
    }
    for step_idx, step in enumerate(env.steps)
    if any(agent.action for agent in step)
]
with open(OUT / "actions.json", "w") as f:
    json.dump(actions, f, indent=2, default=list)
logger.info(f"Actions saved to outputs/actions.json ({len(actions)} steps with moves).")
