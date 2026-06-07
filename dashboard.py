"""
dashboard.py — Live training dashboard for Dreamer v3.
Import in the training notebook: from dashboard import TrainingDashboard, play_with_dashboard
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ipywidgets as widgets
from IPython.display import display, HTML
from collections import deque
from PIL import Image
from dreamer import symexp

ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B']
CKPT_COLORS  = ['#58a6ff','#3fb950','#f0883e','#d2a8ff','#ffa657','#ff7b72','#79c0ff','#56d364']

plt.rcParams.update({
    'figure.facecolor':'#0d1117','axes.facecolor':'#161b22',
    'text.color':'#c9d1d9','axes.edgecolor':'#30363d',
    'xtick.color':'#8b949e','ytick.color':'#8b949e',
    'grid.color':'#21262d','grid.linewidth':0.5,
    'axes.titlecolor':'#58a6ff','axes.labelcolor':'#8b949e',
})

# ─────────────────────────────────────────────────────────────────────────────
class TrainingDashboard:
    MAXLEN     = 2000
    CKPT_EVERY = 3       # must match training loop

    def __init__(self, num_envs, dashboard_every=25):
        self.num_envs       = num_envs
        self.dashboard_every = dashboard_every

        # metric history
        self.ep_rew  = deque(maxlen=self.MAXLEN)
        self.wm      = deque(maxlen=self.MAXLEN)
        self.recon   = deque(maxlen=self.MAXLEN)
        self.kl      = deque(maxlen=self.MAXLEN)
        self.rew_l   = deque(maxlen=self.MAXLEN)
        self.goal    = deque(maxlen=self.MAXLEN)
        self.actor   = deque(maxlen=self.MAXLEN)
        self.critic  = deque(maxlen=self.MAXLEN)
        self.entropy = deque(maxlen=self.MAXLEN)
        self.adv     = deque(maxlen=self.MAXLEN)

        # state
        self.best_dream  = None
        self.episode     = 0
        self.grad_steps  = 0
        self.env_steps   = 0
        self.play_steps  = 0
        self.phase       = 'init'
        self.inner_step  = 0
        self.inner_total = 0
        self.last_score  = 0.0
        self.hero_th     = 0.0

        # per-agent reward tracking
        self.agent_cur   = [0.0] * num_envs
        self.agent_hist  = [deque(maxlen=600) for _ in range(num_envs)]

        # widgets
        self.out_hdr    = widgets.Output()
        self.out_charts = widgets.Output()
        self.out_dream  = widgets.Output()
        self.out_agents = widgets.Output()
        self.out_log    = widgets.Output(layout=widgets.Layout(
                            height='130px', overflow_y='auto',
                            border='1px solid #d0d7de', border_radius='6px',
                            padding='4px'))

        display(HTML(self._css()))
        display(widgets.VBox([
            self.out_hdr,
            self.out_charts,
            self.out_dream,
            widgets.HTML('<div style="color:#58a6ff;font-size:12px;font-weight:600;'
                         'padding:4px 8px;margin-top:6px">🎮 Agent Reward Streams</div>'),
            self.out_agents,
            widgets.HTML('<div style="color:#58a6ff;font-size:12px;font-weight:600;'
                         'padding:4px 8px;margin-top:6px">📋 Training Log</div>'),
            self.out_log,
        ], layout=widgets.Layout(width='100%')))

    # ── CSS ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _css():
        return """<style>
        .dh{background:linear-gradient(135deg,#1c2333,#0d1117);border-radius:12px;
            padding:14px 20px;border:1px solid #30363d;margin-bottom:8px}
        .kpi{display:inline-block;background:#161b22;border:1px solid #30363d;
            border-radius:8px;padding:8px 16px;margin:4px;min-width:120px;text-align:center}
        .kl{font-size:10px;color:#8b949e;font-family:monospace}
        .kv{font-size:19px;font-weight:700;font-family:monospace}
        </style>"""

    # ── Header ───────────────────────────────────────────────────────────────
    def _render_header(self):
        col = {'init':'#8b949e','playing':'#3fb950','training':'#58a6ff',
               'saving':'#e3b341','done':'#56d364'}.get(self.phase,'#8b949e')

        # checkpoint countdown
        if self.episode > 0:
            ep_in_cycle = ((self.episode - 1) % self.CKPT_EVERY) + 1
        else:
            ep_in_cycle = 0

        kpis = [
            ('Episode',              f'{self.episode:,}',                   '#58a6ff'),
            ('Env Steps',            f'{self.env_steps:,}',                 '#c9d1d9'),
            ('Grad Steps',           f'{self.grad_steps:,}',                '#c9d1d9'),
            ('Last Score',           f'{self.last_score:.2f}',              '#3fb950'),
            ('Hero Thresh',          f'{self.hero_th:.2f}',                 '#e3b341'),
            ('Ep to Checkpoint',     f'{ep_in_cycle}/{self.CKPT_EVERY}',    '#d2a8ff'),
        ]
        kpi_html = ''.join(
            f'<div class="kpi"><div class="kl">{l}</div>'
            f'<div class="kv" style="color:{c}">{v}</div></div>'
            for l, v, c in kpis)
        ckpt_html = ''

        # phase bar
        if self.phase == 'training' and self.inner_total:
            pct = int(100 * self.inner_step / self.inner_total)
            bar = '█'*int(pct/5) + '░'*(20-int(pct/5))
            phase_html = (f'<div style="margin-top:8px;font-family:monospace;font-size:11px;color:#8b949e">'
                          f'<span style="color:{col}">[TRAINING]</span>  {bar} {pct}%'
                          f'  ({self.inner_step}/{self.inner_total})</div>')
        elif self.phase == 'playing':
            phase_html = (f'<div style="margin-top:8px;font-family:monospace;font-size:11px;color:#8b949e">'
                          f'<span style="color:#3fb950">[PLAYING]</span>  '
                          f'🎮 Agent exploring…  '
                          f'<b style="color:#c9d1d9">{self.play_steps:,}</b> env steps so far</div>')
        elif self.phase == 'saving':
            phase_html = ('<div style="margin-top:8px;font-family:monospace;font-size:11px;color:#e3b341">'
                          '[SAVING] 💾 Writing checkpoint…</div>')
        else:
            phase_html = ''

        html = (f'<div class="dh">'
                f'<h2 style="color:#58a6ff;margin:0 0 8px;font-family:Segoe UI,sans-serif">'
                f'🧠 Dreamer v3 — Live Training Dashboard</h2>'
                f'{kpi_html}{ckpt_html}{phase_html}</div>')
        with self.out_hdr:
            self.out_hdr.clear_output(wait=True)
            display(HTML(html))

    # ── Metric charts ────────────────────────────────────────────────────────
    def _render_charts(self):
        if len(self.wm) < 2:
            return
        fig = plt.figure(figsize=(16, 7), facecolor='#0d1117')
        gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.38,
                                left=0.05, right=0.97, top=0.93, bottom=0.08)

        def p(ax, data, lbl, col):
            xs, ys = np.arange(len(data)), np.array(data)
            ax.fill_between(xs, ys, alpha=0.18, color=col)
            ax.plot(xs, ys, color=col, linewidth=1.2)
            ax.set_title(lbl, fontsize=9, pad=4)
            ax.grid(True); ax.tick_params(labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor('#30363d')

        specs = [
            (gs[0,0], self.ep_rew, 'Episode Reward',    '#3fb950'),
            (gs[0,1], self.wm,     'World Model Loss',  '#58a6ff'),
            (gs[0,2], self.recon,  'Recon Loss',        '#d2a8ff'),
            (gs[0,3], self.kl,     'KL Loss',           '#f0883e'),
            (gs[1,0], self.actor,  'Actor Loss',        '#ffa657'),
            (gs[1,1], self.critic, 'Critic Loss',       '#ff7b72'),
            (gs[1,2], self.entropy,'Entropy',           '#79c0ff'),
            (gs[1,3], self.adv,    'Advantages',        '#56d364'),
        ]
        for spec, data, lbl, col in specs:
            p(fig.add_subplot(spec), data, lbl, col)

        plt.suptitle('Training Metrics', color='#c9d1d9', fontsize=11, y=0.99)
        with self.out_charts:
            self.out_charts.clear_output(wait=True)
            plt.show()
        plt.close(fig)

    # ── Best dream ───────────────────────────────────────────────────────────
    def _render_dream(self, dreamer_obj):
        if self.best_dream is None:
            return
        rv, bs, br, bv, ba = self.best_dream
        with torch.no_grad():
            dec = symexp(dreamer_obj.decoder(bs.to(dreamer_obj.device))).cpu().numpy()
        H = dec.shape[0]
        fig, axes = plt.subplots(1, H, figsize=(H*1.7, 2.6), facecolor='#0d1117')
        if H == 1: axes = [axes]
        for i in range(H):
            img = np.clip(np.transpose(dec[i], (1,2,0)), 0, 1)
            pil = Image.fromarray((img*255).astype(np.uint8)).resize((96,96), Image.NEAREST)
            axes[i].imshow(np.array(pil)); axes[i].axis('off')
            if i < H-1:
                ai  = int(torch.argmax(ba[i]).item())
                lbl = ACTION_NAMES[ai] if ai < len(ACTION_NAMES) else str(ai)
                axes[i].set_title(f'{lbl}\nr:{br[i].item():.2f}\nv:{bv[i].item():.2f}',
                                  fontsize=6.5, color='#c9d1d9', pad=2)
            else:
                axes[i].set_title(f'v:{bv[i].item():.2f}\nEnd',
                                  fontsize=6.5, color='#8b949e', pad=2)
        plt.suptitle(f'🌙 Best Dream  (Σ reward = {rv:.2f})',
                     color='#58a6ff', fontsize=10, y=1.01)
        plt.tight_layout()
        with self.out_dream:
            self.out_dream.clear_output(wait=True)
            plt.show()
        plt.close(fig)

    # ── Per-agent reward streams ──────────────────────────────────────────────
    def _render_agents(self):
        fig, ax = plt.subplots(figsize=(16, 2.4), facecolor='#0d1117')
        ax.set_facecolor('#161b22')
        has = False
        for i in range(self.num_envs):
            if len(self.agent_hist[i]) > 1:
                ys = np.array(self.agent_hist[i])
                ax.plot(ys, color=CKPT_COLORS[i % len(CKPT_COLORS)],
                        linewidth=1.4, label=f'Env {i+1}')
                has = True
        if has:
            ax.legend(loc='upper left', fontsize=8, fancybox=True, framealpha=0.25,
                      labelcolor='#c9d1d9', facecolor='#161b22', edgecolor='#30363d')
        ax.set_title('Per-Agent Cumulative Reward — current play episode',
                     fontsize=9, color='#8b949e', pad=4)
        ax.grid(True); ax.tick_params(labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
        with self.out_agents:
            self.out_agents.clear_output(wait=True)
            plt.show()
        plt.close(fig)

    # ── Styled log ───────────────────────────────────────────────────────────
    def log(self, msg, level='info'):
        col = {'info':'#57606a','good':'#1a7f37','warn':'#9a6700',
               'hero':'#bc4c00','save':'#8250df'}.get(level, '#57606a')
        with self.out_log:
            display(HTML(
                f'<div style="font-family:monospace;font-size:11px;'
                f'color:{col};padding:1px 4px">{msg}</div>'))

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def on_grad_step(self, wm, dm, bdd, dreamer_obj):
        self.wm.append(wm.get('world_model_loss',0))
        self.recon.append(wm.get('reconstruction_loss',0))
        self.kl.append(wm.get('kl_loss',0))
        self.rew_l.append(wm.get('reward_loss',0))
        self.goal.append(wm.get('goal_loss',0))
        self.actor.append(dm.get('actor_loss',0))
        self.critic.append(dm.get('critic_loss',0))
        self.entropy.append(dm.get('entropies',0))
        self.adv.append(dm.get('advantages',0))

        if self.best_dream is None or bdd[0] > self.best_dream[0]:
            mv, st, rw, vl, ac = bdd
            self.best_dream = (mv, st.detach().cpu(), rw.detach().cpu(),
                               vl.detach().cpu(), ac.detach().cpu())
        self.grad_steps += 1
        if self.grad_steps % self.dashboard_every == 0:
            self._render_header()
            self._render_charts()
            self._render_dream(dreamer_obj)

    def on_env_step(self, env_idx, reward, total_env_steps):
        self.agent_cur[env_idx]  += reward
        self.agent_hist[env_idx].append(self.agent_cur[env_idx])
        self.play_steps = total_env_steps
        self.env_steps  = total_env_steps

    def on_env_reset(self, env_idx):
        self.agent_cur[env_idx] = 0.0
        self.agent_hist[env_idx].clear()

    def on_episode_end(self, score, hero_th, env_steps, dreamer_obj):
        self.ep_rew.append(score)
        self.last_score = score
        self.hero_th    = hero_th
        self.env_steps  = env_steps
        self.play_steps = 0
        self.best_dream = None
        self._render_header()
        lvl = 'good' if score >= hero_th else 'info'
        self.log(f'[Ep {self.episode}] score={score:.2f}  hero_th={hero_th:.2f}  '
                 f'env_steps={env_steps:,}  grad_steps={self.grad_steps:,}', lvl)


# ─────────────────────────────────────────────────────────────────────────────
def play_with_dashboard(dreamer, db, number_of_episodes_per_env=1, hdr_every=50):

    d = dreamer
    num_envs = len(d.envs)
    eps_done  = [0]  * num_envs
    cur_rew   = [0.0]* num_envs
    scores    = []
    local_buf = [[] for _ in range(num_envs)]

    kv_cache = None
    seq_lens = torch.zeros(num_envs, device=d.device, dtype=torch.long)

    obs_list, ram_list = [], []
    for i, env in enumerate(d.envs):
        o, info = env.reset()
        obs_list.append(o)
        ram_list.append(np.array(info['milestones'], dtype=np.float32))
        db.on_env_reset(i)

    step_count = 0
    while min(eps_done) < number_of_episodes_per_env:
        obs_t = (torch.from_numpy(np.array(obs_list)).float() / 255.0).to(d.device)
        ram_t = torch.from_numpy(np.array(ram_list)).float().to(d.device)

        with torch.no_grad():
            ei = d.image_encoder(obs_t)
            eg = d.goal_encoder(ram_t)
            
            seq_lens += 1
            h, kv_cache = d.recurrentModel(
                None, z, a, 
                kv_cache=kv_cache, seq_lens=seq_lens
            )
            z, _ = d.posteriorNet(torch.cat((h, ei, eg), -1))
            a, _, _ = d.actor(torch.cat((h, z), -1))

        a_idx = torch.argmax(a, dim=-1).cpu().numpy()
        a_buf = a.cpu().numpy().astype(np.float32)

        for i, env in enumerate(d.envs):
            if eps_done[i] < number_of_episodes_per_env:
                next_o, r, term, trunc, next_info = env.step(a_idx[i])
                done = term or trunc
                d.total_num_steps += 1
                cur_rew[i] += r
                next_ram = np.array(next_info['milestones'], dtype=np.float32)
                local_buf[i].append((obs_list[i].copy(), ram_list[i].copy(),
                                     a_buf[i].copy(), r))
                db.on_env_step(i, r, d.total_num_steps)
                obs_list[i] = next_o
                ram_list[i] = next_ram

                if done:
                    ep_r = cur_rew[i]
                    for t in local_buf[i]: d.buffer.add(*t)
                    is_empty = (not d.hero_buffer.full and d.hero_buffer.index == 0)
                    if ep_r >= d.hero_threshold or is_empty:
                        for t in local_buf[i]: d.hero_buffer.add(*t)
                        db.log(f'  [!] HERO Env {i+1}  reward={ep_r:.2f}'
                               f'  (thresh={d.hero_threshold:.2f})', 'hero')
                        d.hero_threshold = (ep_r if is_empty
                                            else 0.5*d.hero_threshold + 0.5*ep_r)
                    d.hero_threshold *= d.hero_decay
                    local_buf[i].clear()
                    scores.append(ep_r); d.total_num_episodes += 1; eps_done[i] += 1
                    cur_rew[i] = 0.0
                    if eps_done[i] < number_of_episodes_per_env:
                        no, ni = env.reset()
                        obs_list[i] = no
                        ram_list[i] = np.array(ni['milestones'], dtype=np.float32)
                        db.on_env_reset(i)
                        seq_lens[i] = 0
                        z[i] = torch.zeros(d.latent_dim,    device=d.device)
                        a[i] = torch.zeros(d.action_dim,    device=d.device)

        step_count += 1
        if step_count % hdr_every == 0:
            db._render_header()
            db._render_agents()

    # flush incomplete episodes
    for i in range(num_envs):
        for t in local_buf[i]: d.buffer.add(*t)

    return round(sum(scores)/len(scores), 2) if scores else 0.0
