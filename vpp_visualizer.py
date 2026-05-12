"""
VPP Pipeline Schedule + Memory Peak Visualizer (Final Version)

用法:
    python vpp_visualizer.py --pp 4 --vpp 2 --m 16 --num-layers 8 --output vpp_schedule.png
    python vpp_visualizer.py --pp 4 --vpp 1 --m 8 --act-per-layer 1.0  # 1F1B 对比
"""

import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'


# ─────────────────────────────────────────────
# 1. 调度逻辑(对应 Megatron 源码)
# ─────────────────────────────────────────────

# 反向计算量是前向的 2 倍
BW_COST_MULTIPLIER = 2
FW_COST = 1
BW_COST = BW_COST_MULTIPLIER * FW_COST

# 阶段常量
PHASE_WARMUP = 'warmup'
PHASE_STEADY = 'steady'
PHASE_COOLDOWN = 'cooldown'
# 阶段标记样式 —— 正下方大号标记
PHASE_MARKERS = {
    PHASE_WARMUP:   {'marker': '▲', 'color': '#E8A87C', 'size': 9},   # 灌满: 上三角 (forward)
    PHASE_STEADY:   {'marker': '◆', 'color': '#95B8D1', 'size': 9},   # 交替: 菱形 (F/B配对)
    PHASE_COOLDOWN: {'marker': '▼', 'color': '#D4A0A0', 'size': 9},   # 排空: 下三角 (backward)
}

def get_num_warmup_microbatches(pp_size, pp_rank, vpp_size, total_num_microbatches,
                                 microbatch_group_size, forward_only=False):
    """对应 get_pp_rank_microbatches 中的 VPP 分支"""
    if forward_only:
        return total_num_microbatches
    if vpp_size is None or vpp_size == 1:
        num_warmup = pp_size - pp_rank - 1
    else:
        num_warmup = (pp_size - pp_rank - 1) * 2
        num_warmup += (vpp_size - 1) * microbatch_group_size
    return min(num_warmup, total_num_microbatches)


def get_schedule_table(num_microbatches, num_model_chunks, microbatch_group_size):
    """对应 get_schedule_table — 全局 step → (mb_id, chunk_id)"""
    schedule_table = []
    for min_mb in range(0, num_microbatches, microbatch_group_size):
        upper = min(min_mb + microbatch_group_size, num_microbatches)
        schedule_table.extend([
            (mb_id, chunk_id)
            for chunk_id in range(num_model_chunks)
            for mb_id in range(min_mb, upper)
        ])
    return schedule_table


def validate_config(pp_size, vpp_size, num_microbatches, microbatch_group_size):
    """对应源码里的两个合法性校验"""
    if vpp_size is not None and vpp_size > 1:
        if microbatch_group_size > num_microbatches:
            raise ValueError(
                f"microbatch_group_size_per_vp_stage={microbatch_group_size} "
                f"must be <= num_microbatches={num_microbatches}"
            )
        if microbatch_group_size < pp_size:
            raise ValueError(
                f"microbatch_group_size_per_vp_stage={microbatch_group_size} "
                f"must be >= pipeline_parallel_size={pp_size}"
            )
        final = num_microbatches % microbatch_group_size
        if 0 < final < pp_size:
            raise RuntimeError(
                f"num_microbatches % microbatch_group_size = {final}, "
                f"应该为 0 或 >= {pp_size}, 否则会引入额外 bubble"
            )


# ─────────────────────────────────────────────
# 2. 依赖驱动的事件调度模拟器
# ─────────────────────────────────────────────

class Event:
    """单个 F 或 B 事件"""
    __slots__ = ('rank', 'mb', 'chunk', 'kind', 'order', 'time', 'duration', 'phase')

    def __init__(self, rank, mb, chunk, kind, order, phase):
        self.rank = rank
        self.mb = mb
        self.chunk = chunk
        self.kind = kind          # 'F' or 'B'
        self.order = order        # 该 rank 上的执行顺序
        self.time = None          # 全局起始时刻
        self.duration = BW_COST if kind == 'B' else FW_COST
        self.phase = phase        # 'warmup' / 'steady' / 'cooldown'

    def end_time(self):
        """事件结束时刻"""
        if self.time is None:
            return None
        return self.time + self.duration - 1

    def occupies_time_steps(self):
        """返回该事件占用的所有时间步"""
        if self.time is None:
            return set()
        return set(range(self.time, self.time + self.duration))

    def key(self):
        return (self.rank, self.mb, self.chunk, self.kind)

    def __repr__(self):
        return f"{self.kind}(r{self.rank},mb{self.mb},c{self.chunk})@{self.time}[{self.phase}]"


def build_event_list(pp_size, pp_rank, vpp_size, num_microbatches,
                      microbatch_group_size, mode='1f1b'):
    """
    为单个 rank 构建按执行顺序排列的事件列表

    每个事件标记其所属阶段:
    - warmup:  纯 forward (k < num_warmup)
    - steady:  1F1B 配对 (k ∈ [num_warmup, num_warmup + num_remaining))
    - cooldown: 纯 backward (k >= num_warmup + num_remaining)
    """
    total_mb = num_microbatches * vpp_size
    schedule_table = get_schedule_table(num_microbatches, vpp_size, microbatch_group_size)

    if mode == 'gpipe':
        if vpp_size != 1:
            raise ValueError("GPipe 模式仅支持 V=1")
        num_warmup = total_mb
        num_remaining = 0
    else:
        num_warmup = get_num_warmup_microbatches(
            pp_size, pp_rank, vpp_size, total_mb, microbatch_group_size
        )
        num_remaining = total_mb - num_warmup

    events = []
    order = 0

    # ===== Warmup: 纯 forward =====
    for k in range(num_warmup):
        mb_id, chunk_id = schedule_table[k]
        events.append(Event(pp_rank, mb_id, chunk_id, 'F', order, PHASE_WARMUP))
        order += 1

    # ===== Steady: 1F1B =====
    for k in range(num_remaining):
        # Forward (来自 schedule_table 中 warmup 之后的 entry)
        f_k = k + num_warmup
        f_mb, f_chunk = schedule_table[f_k]
        events.append(Event(pp_rank, f_mb, f_chunk, 'F', order, PHASE_STEADY))
        order += 1

        # Backward (对应 schedule_table 中第 k 个 entry, chunk 翻转)
        b_k = k
        b_mb, b_chunk_fwd = schedule_table[b_k]
        b_chunk = vpp_size - 1 - b_chunk_fwd
        events.append(Event(pp_rank, b_mb, b_chunk, 'B', order, PHASE_STEADY))
        order += 1

    # ===== Cooldown: 纯 backward =====
    for k in range(num_remaining, total_mb):
        b_mb, b_chunk_fwd = schedule_table[k]
        b_chunk = vpp_size - 1 - b_chunk_fwd
        events.append(Event(pp_rank, b_mb, b_chunk, 'B', order, PHASE_COOLDOWN))
        order += 1

    return events


def schedule_global_time(all_events_by_rank, pp_size, vpp_size):
    """
    依赖驱动的事件调度: 按拓扑顺序为每个事件分配全局起始时刻

    依赖规则:
    1. 同一 rank 上: event.order > prev.order ⇒ event.time >= prev.end_time() + 1
    2. F(r, mb, c) 依赖 F(r-1, mb, c) 完成 (r>0)
    3. F(0, mb, c) 依赖 F(P-1, mb, c-1) 完成 (c>0)
    4. B(r, mb, c) 依赖 B(r+1, mb, c) 完成 (r<P-1)
    5. B(P-1, mb, c) 依赖 B(0, mb, c+1) 完成 (c<V-1)
    6. B(P-1, mb, V-1) 依赖 F(P-1, mb, V-1) 完成
    """
    event_index = {}
    for rank, evs in enumerate(all_events_by_rank):
        for ev in evs:
            event_index[ev.key()] = ev

    def deps(ev):
        """返回 ev 的所有前置依赖事件"""
        result = []
        rank, mb, chunk, kind = ev.rank, ev.mb, ev.chunk, ev.kind

        # 规则 1: 同 rank 前一个事件
        if ev.order > 0:
            prev = all_events_by_rank[rank][ev.order - 1]
            result.append(prev)

        if kind == 'F':
            if rank > 0:
                k = (rank - 1, mb, chunk, 'F')
                if k in event_index:
                    result.append(event_index[k])
            else:
                if chunk > 0:
                    k = (pp_size - 1, mb, chunk - 1, 'F')
                    if k in event_index:
                        result.append(event_index[k])
        else:  # 'B'
            if rank < pp_size - 1:
                k = (rank + 1, mb, chunk, 'B')
                if k in event_index:
                    result.append(event_index[k])
            else:
                if chunk < vpp_size - 1:
                    k = (0, mb, chunk + 1, 'B')
                    if k in event_index:
                        result.append(event_index[k])
                else:
                    k = (pp_size - 1, mb, vpp_size - 1, 'F')
                    if k in event_index:
                        result.append(event_index[k])

        return result

    all_events = list(event_index.values())
    unresolved = set(ev.key() for ev in all_events)

    max_iter = len(all_events) * 10
    iters = 0
    while unresolved and iters < max_iter:
        iters += 1
        progress = False
        for ev in all_events:
            if ev.key() not in unresolved:
                continue
            dep_end_times = []
            ok = True
            for d in deps(ev):
                if d.time is None:
                    ok = False
                    break
                dep_end_times.append(d.end_time())
            if not ok:
                continue
            ev.time = (max(dep_end_times) + 1) if dep_end_times else 0
            unresolved.remove(ev.key())
            progress = True
        if not progress:
            raise RuntimeError(f"调度死锁: 还有 {len(unresolved)} 个事件无法满足依赖")

    return all_events


# ─────────────────────────────────────────────
# 3. Bubble rate 统计
# ─────────────────────────────────────────────

def compute_bubble_rate(events_by_rank, pp_size, max_time):
    """
    从模拟结果统计实际 bubble rate
    每个事件可占据多个连续时间步
    """
    total_slots = max_time + 1
    per_rank_bubble = []
    total_bubble_cells = 0
    total_busy_cells = 0

    for rank in range(pp_size):
        busy_times = set()
        for ev in events_by_rank[rank]:
            busy_times.update(ev.occupies_time_steps())
        busy = len(busy_times)
        bubble = total_slots - busy
        per_rank_bubble.append(bubble / total_slots)
        total_busy_cells += busy
        total_bubble_cells += bubble

    total_cells = pp_size * total_slots
    system_bubble = total_bubble_cells / total_cells

    return {
        'per_rank': per_rank_bubble,
        'system': system_bubble,
        'total_bubble_cells': total_bubble_cells,
        'total_busy_cells': total_busy_cells,
        'total_cells': total_cells,
        'total_slots': total_slots,
    }


def theoretical_bubble_rate(pp_size, vpp_size, num_microbatches):
    """理论 bubble rate 公式"""
    V = vpp_size if vpp_size is not None else 1
    approx = (pp_size - 1) / (V * num_microbatches)
    strict = (pp_size - 1) / (V * num_microbatches + pp_size - 1)
    return approx, strict


# ─────────────────────────────────────────────
# 4. 显存峰值计算（以 μ(s) 公式修正）
# ─────────────────────────────────────────────

def compute_memory_peaks(pp_size, vpp_size, num_microbatches,
                          microbatch_group_size, act_per_layer, layers_per_device,
                          mode='1f1b'):
    """
    显存峰值计算
    核心公式:
      VPP:  μ(s) = P·(V-1) + 2·(P-s-1) + 1  (当 N=P 时)
      1F1B: μ(s) = P - s
    相邻 stage 差 2 (VPP) 或 1 (1F1B)
    """
    layers_per_chunk = layers_per_device / vpp_size
    act_per_chunk = act_per_layer * layers_per_chunk

    peaks = []
    warmups = []
    in_flight_counts = []
    bpipe_mu = []

    for pp_rank in range(pp_size):
        if vpp_size >= 2:
            mu = pp_size * (vpp_size - 1) + 2 * (pp_size - pp_rank - 1) + 1
            bpipe_mu.append(mu)
            num_warmup = mu - 1
            in_flight = mu
        else:
            mu = pp_size - pp_rank
            bpipe_mu.append(mu)
            num_warmup = pp_size - pp_rank - 1
            in_flight = mu

        if mode == 'gpipe':
            in_flight = num_microbatches * vpp_size
            num_warmup = in_flight - 1

        peaks.append(in_flight * act_per_chunk)
        warmups.append(num_warmup)
        in_flight_counts.append(in_flight)

    return peaks, warmups, in_flight_counts, bpipe_mu


# ─────────────────────────────────────────────
# 5. 绘图
# ─────────────────────────────────────────────

COLORS = {
    'F_chunk': [
        '#B5D4F4', '#CECBF6', '#9FE1CB', '#FAC775',
        '#F4C0D1', '#C0DD97',
    ],
    'B_chunk': [
        '#F09595', '#F5C4B3', '#85B7EB', '#EF9F27',
        '#ED93B1', '#97C459',
    ],
    'F_edge': '#378ADD',
    'B_edge': '#D85A30',
    'bubble': '#F1EFE8',
    'bubble_edge': '#D3D1C7',
}


def plot_schedule(ax, events_by_rank, pp_size, vpp_size, max_time):
    """画依赖驱动的甘特图（B 占 2 格宽，正下方大号标记区分三阶段）"""
    row_h = 0.78
    cell_w = 0.92
    pad = 0.04

    # 占用记录
    occupied = defaultdict(set)
    for rank, events in enumerate(events_by_rank):
        for ev in events:
            occupied[rank].update(ev.occupies_time_steps())

    for rank in range(pp_size):
        y = pp_size - rank - 1

        # bubble 格
        for t in range(max_time + 1):
            if t in occupied[rank]:
                continue
            rect = FancyBboxPatch(
                (t + pad, y + pad), cell_w - 2 * pad, row_h - 2 * pad,
                boxstyle="round,pad=0.01",
                facecolor=COLORS['bubble'], edgecolor=COLORS['bubble_edge'],
                linewidth=0.4, zorder=1
            )
            ax.add_patch(rect)

        # F/B 格
        for ev in events_by_rank[rank]:
            x = ev.time
            duration = ev.duration
            if ev.kind == 'F':
                c = ev.chunk % len(COLORS['F_chunk'])
                color = COLORS['F_chunk'][c]
                edge_color = COLORS['F_edge']
                txt = f"F{ev.mb}\nc{ev.chunk}"
            else:
                c = ev.chunk % len(COLORS['B_chunk'])
                color = COLORS['B_chunk'][c]
                edge_color = COLORS['B_edge']
                txt = f"B{ev.mb}\nc{ev.chunk}"

            rect_width = cell_w * duration - 2 * pad
            rect = FancyBboxPatch(
                (x + pad, y + pad), rect_width, row_h - 2 * pad,
                boxstyle="round,pad=0.01",
                facecolor=color, edgecolor=edge_color,
                linewidth=0.5, zorder=3
            )
            ax.add_patch(rect)

            fontsize = max(5, min(7, 90 // max(max_time, 1)))
            text_x = x + duration * 0.5
            ax.text(text_x, y + row_h / 2 + 0.06, txt,
                    ha='center', va='center', fontsize=fontsize,
                    color='#2C2C2A', zorder=4)

            # 正下方阶段标记
            marker_info = PHASE_MARKERS[ev.phase]
            if marker_info['marker']:
                marker_x = x + duration * 0.5
                marker_y = y - 0.02
                ax.text(marker_x, marker_y, marker_info['marker'],
                        ha='center', va='top', fontsize=marker_info['size'],
                        color=marker_info['color'], zorder=5,
                        fontweight='bold')

    ax.set_xlim(-0.2, max_time + 1.2)
    ax.set_ylim(-0.35, pp_size + 0.1)
    ax.set_yticks([pp_size - r - 0.5 for r in range(pp_size)])
    ax.set_yticklabels([f'rank {r}' for r in range(pp_size)], fontsize=9)
    ax.set_xlabel('global time step (event-driven schedule, B=2xF)', fontsize=9)
    step = max(1, (max_time + 1) // 20)
    ax.set_xticks(range(0, max_time + 2, step))
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='x', alpha=0.18, zorder=0)
    ax.set_axisbelow(True)

    # 图例
    legend_elements = []
    for c in range(min(vpp_size, len(COLORS['F_chunk']))):
        legend_elements.append(mpatches.Patch(
            facecolor=COLORS['F_chunk'][c], edgecolor=COLORS['F_edge'],
            linewidth=0.8, label=f'F chunk{c}'))
    for c in range(min(vpp_size, len(COLORS['B_chunk']))):
        legend_elements.append(mpatches.Patch(
            facecolor=COLORS['B_chunk'][c], edgecolor=COLORS['B_edge'],
            linewidth=0.8, label=f'B chunk{c}'))
    legend_elements.append(mpatches.Patch(
        facecolor=COLORS['bubble'], edgecolor=COLORS['bubble_edge'],
        linewidth=0.8, label='bubble'))
    # 阶段图例
    from matplotlib.lines import Line2D
    legend_elements.append(Line2D([0], [0], marker='s', color='w',
                                   markerfacecolor='white', markeredgecolor='black',
                                   markersize=10, label='warmup (▲)'))
    legend_elements.append(Line2D([0], [0], marker='s', color='w',
                                   markerfacecolor='white', markeredgecolor='black',
                                   markersize=10, label='steady (◆)'))
    legend_elements.append(Line2D([0], [0], marker='s', color='w',
                                   markerfacecolor='white', markeredgecolor='black',
                                   markersize=10, label='cooldown (▼)'))
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=7, ncol=min(len(legend_elements), 6),
              framealpha=0.95)


def plot_memory(ax, peaks_vpp, peaks_pp, in_flight_vpp, in_flight_pp,
                 bpipe_mu_vpp, pp_size, act_per_layer, layers_per_device, vpp_size):
    """显存峰值对比 + BPipe 理论值标注"""
    x = np.arange(pp_size)
    width = 0.35

    bars_pp = ax.bar(x - width / 2, peaks_pp, width,
                      label='1F1B (V=1)',
                      color='#B5D4F4', edgecolor='#378ADD', linewidth=0.8)
    bars_vpp = ax.bar(x + width / 2, peaks_vpp, width,
                       label=f'VPP (V={vpp_size})',
                       color='#CECBF6', edgecolor='#534AB7', linewidth=0.8)

    for bar, inflight in zip(bars_pp, in_flight_pp):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03,
                f'{h:.2f}\n(n={inflight})',
                ha='center', va='bottom', fontsize=7, color='#185FA5')

    for i, (bar, inflight) in enumerate(zip(bars_vpp, in_flight_vpp)):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03,
                f'{h:.2f}\n(n={inflight}, mu={bpipe_mu_vpp[i]})',
                ha='center', va='bottom', fontsize=7, color='#3C3489')

    ymax = max(max(peaks_pp), max(peaks_vpp)) if peaks_pp and peaks_vpp else 1
    for i in range(pp_size):
        if peaks_pp[i] > 0:
            ratio = peaks_vpp[i] / peaks_pp[i]
            ax.text(i, -ymax * 0.10, f'x{ratio:.2f}',
                    ha='center', va='top', fontsize=8,
                    color='#993C1D', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'rank {r}' for r in range(pp_size)], fontsize=9)
    ax.set_ylabel('activation memory (units)', fontsize=9)
    ax.set_xlabel('pipeline rank', fontsize=9)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(-ymax * 0.18, ymax * 1.25)

    layers_per_chunk = layers_per_device / vpp_size
    ax.set_title(
        f'Activation memory peak  '
        f'(act/layer={act_per_layer}, layers/device={layers_per_device}, '
        f'layers/chunk={layers_per_chunk:.2g})',
        fontsize=9
    )


def plot_per_rank_bubble(ax, bubble_stats, pp_size, theory_approx):
    """每个 rank 的 bubble 占比柱状图"""
    x = np.arange(pp_size)
    rates = [r * 100 for r in bubble_stats['per_rank']]
    bars = ax.bar(x, rates, width=0.5,
                  color='#F1EFE8', edgecolor='#888780', linewidth=0.8)

    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.5,
                f'{r:.1f}%', ha='center', va='bottom', fontsize=8, color='#5F5E5A')

    sys_rate = bubble_stats['system'] * 100
    ax.axhline(sys_rate, color='#993C1D', linestyle='--', linewidth=1,
               label=f'simulated system: {sys_rate:.1f}%')
    th_rate = theory_approx * 100
    ax.axhline(th_rate, color='#185FA5', linestyle=':', linewidth=1,
               label=f'theory (P-1)/(VM): {th_rate:.1f}%')

    ax.set_xticks(x)
    ax.set_xticklabels([f'rank {r}' for r in range(pp_size)], fontsize=9)
    ax.set_ylabel('bubble rate (%)', fontsize=9)
    ax.set_xlabel('pipeline rank', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title('Per-rank bubble rate', fontsize=9)


# ─────────────────────────────────────────────
# 6. 阶段统计
# ─────────────────────────────────────────────

def print_phase_summary(events_by_rank, pp_size):
    """打印每个 rank 的三阶段概要"""
    print(f"\n三阶段概要:")
    print(f"{'rank':<6} {'phase':<12} {'#events':<10} {'time range':<20} {'#F':<6} {'#B':<6}")
    print('-' * 65)
    for rank in range(pp_size):
        phases = {PHASE_WARMUP: [], PHASE_STEADY: [], PHASE_COOLDOWN: []}
        for ev in events_by_rank[rank]:
            phases[ev.phase].append(ev)

        for phase_name in [PHASE_WARMUP, PHASE_STEADY, PHASE_COOLDOWN]:
            evs = phases[phase_name]
            if not evs:
                continue
            n_f = sum(1 for e in evs if e.kind == 'F')
            n_b = sum(1 for e in evs if e.kind == 'B')
            t_min = min(e.time for e in evs)
            t_max = max(e.end_time() for e in evs)
            print(f"  {rank:<4} {phase_name:<12} {len(evs):<10} "
                  f"[{t_min}, {t_max}]{'':<8} {n_f:<6} {n_b:<6}")


def main():
    parser = argparse.ArgumentParser(
        description='VPP Pipeline Schedule + Memory Peak Visualizer'
    )
    parser.add_argument('--pp', type=int, default=4)
    parser.add_argument('--vpp', type=int, default=2)
    parser.add_argument('--m', type=int, default=16, help='num_microbatches')
    parser.add_argument('--act-per-layer', type=float, default=1.0)
    parser.add_argument('--num-layers', type=int, default=None,
                        help='总层数,默认 pp*vpp*2')
    parser.add_argument('--microbatch-group-size', type=int, default=None,
                        help='默认 = pp')
    parser.add_argument('--mode', type=str, default='1f1b',
                        choices=['1f1b', 'gpipe'],
                        help='调度模式: 1f1b (默认, 含 VPP) 或 gpipe')
    parser.add_argument('--output', type=str, default='vpp_schedule.png')
    args = parser.parse_args()

    pp_size = args.pp
    vpp_size = args.vpp
    num_microbatches = args.m
    act_per_layer = args.act_per_layer
    microbatch_group_size = args.microbatch_group_size or pp_size
    num_layers = args.num_layers or (pp_size * vpp_size * 2)
    layers_per_device = num_layers // pp_size

    # 合法性校验
    try:
        validate_config(pp_size, vpp_size, num_microbatches, microbatch_group_size)
    except (ValueError, RuntimeError) as e:
        print(f"配置错误: {e}")
        return

    print(f"配置: pp={pp_size}, vpp={vpp_size}, m={num_microbatches}, "
          f"N={microbatch_group_size}")
    print(f"总层数={num_layers}, 每device层数={layers_per_device}, "
          f"每chunk层数={layers_per_device // max(vpp_size, 1)}")
    print(f"F 持续时间=1, B 持续时间={BW_COST_MULTIPLIER} (xF)")

    # 构建事件列表
    events_by_rank = [
        build_event_list(pp_size, r, vpp_size, num_microbatches,
                         microbatch_group_size, mode=args.mode)
        for r in range(pp_size)
    ]

    # 调度全局时间
    all_events = schedule_global_time(events_by_rank, pp_size, vpp_size)
    max_time = max(ev.end_time() for ev in all_events)

    # 三阶段概要
    print_phase_summary(events_by_rank, pp_size)

    # bubble rate 统计
    bubble_stats = compute_bubble_rate(events_by_rank, pp_size, max_time)
    theory_approx, theory_strict = theoretical_bubble_rate(
        pp_size, vpp_size, num_microbatches
    )

    print(f"\n各 rank 时序 (max_time={max_time}):")
    for r in range(pp_size):
        total_mb = num_microbatches * vpp_size
        num_warmup = get_num_warmup_microbatches(
            pp_size, r, vpp_size, total_mb, microbatch_group_size
        )
        ev_count = len(events_by_rank[r])
        max_t = max(ev.end_time() for ev in events_by_rank[r])
        bubble_r = bubble_stats['per_rank'][r]
        print(f"  rank {r}: num_warmup={num_warmup}, mu={num_warmup+1}, "
              f"events={ev_count}, max_t={max_t}, bubble={bubble_r:.1%}")

    print(f"\nBubble rate:")
    print(f"  实测 (模拟): {bubble_stats['system']:.2%}  "
          f"({bubble_stats['total_bubble_cells']}/{bubble_stats['total_cells']} cells)")
    print(f"  理论 (P-1)/(V*M):       {theory_approx:.2%}  (近似式, M>>P 时准)")
    print(f"  理论 (P-1)/(V*M + P-1): {theory_strict:.2%}  (严格式)")
    delta = abs(bubble_stats['system'] - theory_strict)
    print(f"  实测 vs 严格式差距: {delta:.2%}")

    # 显存峰值
    peaks_vpp, warmups_vpp, in_flight_vpp, bpipe_mu_vpp = compute_memory_peaks(
        pp_size, vpp_size, num_microbatches, microbatch_group_size,
        act_per_layer, layers_per_device, mode=args.mode
    )
    peaks_pp, warmups_pp, in_flight_pp, bpipe_mu_pp = compute_memory_peaks(
        pp_size, 1, num_microbatches, microbatch_group_size,
        act_per_layer, layers_per_device, mode='1f1b'
    )

    print(f"\n显存峰值对比:")
    print(f"{'rank':<6} {'num_warmup':<12} {'mu (in-flight)':<14} "
          f"{'peak(VPP)':<12} {'peak(1F1B)':<12} {'ratio':<8}")
    for r in range(pp_size):
        ratio = peaks_vpp[r] / peaks_pp[r] if peaks_pp[r] > 0 else float('inf')
        print(f"  {r:<4} {warmups_vpp[r]:<12} {in_flight_vpp[r]:<14} "
              f"{peaks_vpp[r]:<12.3f} {peaks_pp[r]:<12.3f} "
              f"x{ratio:<7.3f}")

    # ==================== 绘图 ====================
    fig_w = max(16, (max_time + 2) * 0.55)
    fig = plt.figure(figsize=(fig_w, 11))
    fig.patch.set_facecolor('white')

    ax_schedule = fig.add_axes([0.05, 0.46, 0.93, 0.46])
    plot_schedule(ax_schedule, events_by_rank, pp_size, vpp_size, max_time)

    title = (f'Pipeline Schedule  |  pp={pp_size}, vpp={vpp_size}, '
             f'm={num_microbatches}, N={microbatch_group_size}, '
             f'layers={num_layers}\n'
             f'bubble: simulated={bubble_stats["system"]:.1%}  '
             f'theory (P-1)/(VM)={theory_approx:.1%}  '
             f'theory strict={theory_strict:.1%}')
    ax_schedule.set_title(title, fontsize=11, pad=8)

    # 下半部分
    ax_memory = fig.add_axes([0.05, 0.06, 0.55, 0.32])
    plot_memory(ax_memory, peaks_vpp, peaks_pp, in_flight_vpp, in_flight_pp,
                 bpipe_mu_vpp, pp_size, act_per_layer, layers_per_device, vpp_size)

    ax_bubble = fig.add_axes([0.66, 0.06, 0.31, 0.32])
    plot_per_rank_bubble(ax_bubble, bubble_stats, pp_size, theory_approx)

    plt.savefig(args.output, dpi=140, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\n已保存: {args.output}")
    plt.close()


if __name__ == '__main__':
    main()