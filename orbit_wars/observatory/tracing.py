import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from orbit_wars.observatory.decisions import decisions_snapshot


def _build_action_rows(steps: list[Any]) -> list[dict[str, Any]]:
    recorded_decisions = decisions_snapshot()
    targets_by_signature: dict[tuple[int, int, int, float], list[int]] = defaultdict(list)
    for decision in recorded_decisions:
        signature = (
            decision["player"],
            decision["source_planet_id"],
            decision["ships"],
            decision["angle_rad"],
        )
        targets_by_signature[signature].append(decision["target_planet_id"])

    rows: list[dict[str, Any]] = []
    for step_idx, step in enumerate(steps):
        for agent_idx, agent in enumerate(step):
            if not agent.action:
                continue
            for move in agent.action:
                from_id, angle, ships = move
                rounded_angle = round(angle, 4)
                signature = (agent_idx, from_id, ships, rounded_angle)
                target_candidates = targets_by_signature.get(signature, [])
                target_id = target_candidates.pop(0) if target_candidates else None
                rows.append(
                    {
                        "step": step_idx,
                        "player": agent_idx,
                        "source_planet_id": from_id,
                        "target_planet_id": target_id,
                        "angle_rad": rounded_angle,
                        "ships": ships,
                    }
                )
    return rows


def _render_action_overlay(action_rows: list[dict[str, Any]]) -> str:
    table_rows_html = "\n".join(
        f'<tr data-step="{r["step"]}">'
        f"<td>{r['step']}</td>"
        f"<td class='p{r['player']}'>P{r['player']}</td>"
        f"<td>{r['source_planet_id']}</td>"
        f"<td>{'' if r['target_planet_id'] is None else r['target_planet_id']}</td>"
        f"<td>{r['angle_rad']:.4f}</td>"
        f"<td>{r['ships']}</td>"
        f"</tr>"
        for r in action_rows
    )

    return f"""
<style>
  #action-log {{
    position: fixed; top: 0; right: 0;
    width: 500px; max-height: 560px;
    background: rgba(15,15,25,0.92); color: #e0e0e0;
    font: 12px/1.45 monospace; border-bottom-left-radius: 8px;
    overflow-y: auto; z-index: 9999; padding: 6px 8px;
    transition: max-height 0.2s ease, width 0.2s ease;
  }}
  #action-log.collapsed {{
    max-height: 34px;
    overflow: hidden;
  }}
  #action-log h3 {{
    margin: 0 0 4px; font-size: 13px; color: #aaf; text-align: center;
    position: sticky; top: 0; background: rgba(15,15,25,0.95); padding: 4px 0;
  }}
  #action-log .toolbar {{
    display: flex; justify-content: flex-end; gap: 6px; margin: 0 0 4px;
    position: sticky; top: 21px; background: rgba(15,15,25,0.95); padding-bottom: 3px;
  }}
  #action-log button {{
    border: 1px solid #556; border-radius: 4px; background: #222a40;
    color: #d9e1ff; font: 11px monospace; padding: 2px 6px; cursor: pointer;
  }}
  #action-log .toggle-label {{
    display: inline-flex; align-items: center; gap: 4px; color: #d9e1ff;
    font: 11px monospace; border: 1px solid #556; border-radius: 4px; padding: 2px 6px;
    background: #222a40;
  }}
  #action-log table {{ width: 100%; border-collapse: collapse; }}
  #action-log th {{
    color: #88aaff; border-bottom: 1px solid #444; padding: 2px 4px;
    position: sticky; top: 50px; background: rgba(15,15,25,0.95);
  }}
  #action-log td {{ padding: 2px 4px; border-bottom: 1px solid #222; }}
  #action-log .p0 {{ color: #66ccff; }}
  #action-log .p1 {{ color: #ff9966; }}
  #action-log tr.active {{ background: rgba(255,230,80,0.25); outline: 1px solid #ffd700; }}
</style>
<div id="action-log">
  <h3>Actions ({len(action_rows)} moves)</h3>
  <div class="toolbar">
    <label class="toggle-label" for="planet-ids-toggle">
      <input id="planet-ids-toggle" type="checkbox" checked />
      IDs
    </label>
    <label class="toggle-label" for="planet-ships-toggle">
      <input id="planet-ships-toggle" type="checkbox" checked />
      Ships
    </label>
    <button id="action-log-toggle" type="button">Collapse</button>
  </div>
  <table>
    <thead><tr><th>Step</th><th>Who</th><th>Source</th><th>Target</th><th>Angle</th><th>Ships</th></tr></thead>
    <tbody>
      {table_rows_html}
    </tbody>
  </table>
</div>
<script>
(function () {{
  let lastStep = -1;
  let showIds = true;
  let showShips = true;
  const panel = document.getElementById('action-log');
  const toggle = document.getElementById('action-log-toggle');
  const idsToggle = document.getElementById('planet-ids-toggle');
  const shipsToggle = document.getElementById('planet-ships-toggle');

  function updateToggleLabel() {{
    toggle.textContent = panel.classList.contains('collapsed') ? 'Expand' : 'Collapse';
  }}

  toggle.addEventListener('click', function () {{
    panel.classList.toggle('collapsed');
    updateToggleLabel();
  }});
  idsToggle.addEventListener('change', function () {{
    showIds = !!idsToggle.checked;
  }});
  shipsToggle.addEventListener('change', function () {{
    showShips = !!shipsToggle.checked;
  }});
  updateToggleLabel();

  function installRendererLabelHook() {{
    if (!window.kaggle || typeof window.kaggle.renderer !== 'function') return false;
    const originalRenderer = window.kaggle.renderer;
    if (originalRenderer.__orbitWarsLabelsWrapped) return true;

    const wrappedRenderer = async function (context) {{
      await originalRenderer(context);
      const canvas = context?.parent?.querySelector('canvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const obs = context?.environment?.steps?.[context.step]?.[0]?.observation;
      const planets = obs?.planets || [];
      const scale = canvas.width / 100.0;

      ctx.save();
      ctx.font = `${{Math.max(11, (12 * scale) / 4)}}px monospace`;
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';

      planets.forEach(function (planet) {{
        const id = planet[0];
        const x = planet[2] * scale;
        const y = planet[3] * scale;
        const r = planet[4] * scale;
        const ships = Math.floor(planet[5]);
        const idText = String(id);
        const rightPad = 4;
        const rightX = x + r + rightPad;

        if (showIds) {{
          ctx.textAlign = 'center';
          ctx.fillStyle = '#ff4d4d';
          ctx.fillText(idText, x, y);
        }}

        if (showShips) {{
          ctx.textAlign = 'left';
          ctx.fillStyle = '#c9c9c9';
          ctx.fillText('|', rightX, y);
          ctx.fillStyle = '#7dd3fc';
          ctx.fillText(` ${{ships}}`, rightX + ctx.measureText('|').width, y);
        }}
      }});
      ctx.restore();
    }};

    wrappedRenderer.__orbitWarsLabelsWrapped = true;
    window.kaggle.renderer = wrappedRenderer;
    return true;
  }}

  function highlightStep(step) {{
    if (step === lastStep) return;
    lastStep = step;
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
      const m = el.textContent.match(/(\\d+)/);
      if (m) {{ highlightStep(parseInt(m[1])); return; }}
    }}
    const m = document.body.innerText.match(/\\bstep[:\\s]+(\\d+)/i);
    if (m) highlightStep(parseInt(m[1]));
  }}

  new MutationObserver(detectStep).observe(document.body, {{
    subtree: true, childList: true, characterData: true
  }});
  if (!installRendererLabelHook()) {{
    const hookTimer = window.setInterval(function () {{
      if (installRendererLabelHook()) {{
        window.clearInterval(hookTimer);
      }}
    }}, 100);
  }}

  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', detectStep);
  }} else {{
    detectStep();
  }}
}})();
</script>
"""


def _serialize_observation(observation: Any) -> Any:
    if isinstance(observation, dict):
        return observation
    return vars(observation)


def export_run_artifacts(env: Any, output_dir: str | Path = "outputs") -> Path:
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    action_rows = _build_action_rows(env.steps)
    html = env.render(mode="html", width=800, height=600)
    html = re.sub(
        r"drawText\(\s*Math\.floor\(ships\)\.toString\(\)\s*,\s*x\s*,\s*y\s*,\s*'#FFFFFF'\s*,\s*12\s*\);",
        "// Ship count inside planet removed by observatory overlay.",
        html,
        count=1,
    )
    html = html.replace(
        "</body>",
        _render_action_overlay(action_rows) + "\n</body>",
    )

    with (out / "replay.html").open("w") as f:
        f.write(html)

    observations = [
        {
            "step": step_idx,
            "agents": [
                {
                    "player": agent_idx,
                    "reward": agent.reward,
                    "status": agent.status,
                    "action": agent.action,
                    "observation": _serialize_observation(agent.observation),
                }
                for agent_idx, agent in enumerate(step)
            ],
        }
        for step_idx, step in enumerate(env.steps)
    ]
    with (out / "observations.json").open("w") as f:
        json.dump(observations, f, indent=2, default=list)

    with (out / "action_table.json").open("w") as f:
        json.dump(action_rows, f, indent=2, default=list)

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
    with (out / "actions.json").open("w") as f:
        json.dump(actions, f, indent=2, default=list)

    return out
