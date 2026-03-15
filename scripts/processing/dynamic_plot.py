import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import time
from matplotlib.patches import Patch
# 载入数据
dt_l = np.load('/home/czk/left_heel.npy')[::2]
dt_r = np.load('/home/czk/right_heel.npy')[::2]
lto  = np.load('/home/czk/lto.npy')[1::2] + np.load('/home/czk/lto.npy')[::2]
lto[lto!=2000] -= 1000 
rto  = np.load('/home/czk/rto.npy')[1::2] + np.load('/home/czk/rto.npy')[::2]
rto[rto!=2000] -= 1000
lhs  = np.load('/home/czk/lhs.npy')[1::2] + np.load('/home/czk/lhs.npy')[::2]
lhs[lhs!=2000] -= 1000
rhs  = np.load('/home/czk/rhs.npy')[1::2] + np.load('/home/czk/rhs.npy')[::2]
rhs[rhs!=2000] -= 1000
dto = np.copy(lto)
dto[rto!=2000] = rto[rto!=2000]
dhs = np.copy(lhs)
dhs[rhs!=2000] = rhs[rhs!=2000]
print(len(dt_l))

t = np.arange(len(dt_l)) * 9.3 / len(dt_l)

# 可选：SciencePlots 风格包可能没装，做个保护
# try:
#     plt.style.use(['science', 'ieee'])
# except Exception:
#     pass3

plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(figsize=(20, 5))

# 先固定坐标轴范围
ax.set_ylim(-275, 275)
ax.set_xlim(-0.5, 10)
ax.set_xlabel('Time (s)', fontsize=16, labelpad=3)      # X 轴标签
ax.set_ylabel('Angular velocity (°/s)', fontsize=16, labelpad=3)   # Y 轴标签

# 预创建两条空线（颜色与你原代码一致）
line_l, = ax.plot([], [], '-', color='darkorange', lw=4)
line_r, = ax.plot([], [], '-', color='steelblue',  lw=4)
to = ax.scatter([], [], marker='o',facecolors='none',
                 edgecolors='red',s=60, linewidths=1.5,zorder=3)
hs = ax.scatter([], [], marker='s',facecolors='none',
                 edgecolors='black',s=60, linewidths=1.5,zorder=3)
# ax.legend(loc='upper right')
ms_to = np.sqrt(60)   # s=60 -> markersize≈sqrt(60)
ms_hs = np.sqrt(60)

def make_span_patches(ax, intervals, color='#82c8a0', alpha=0.25):
    """为每段区间创建一个竖直条带（满高），初始 width=0。"""
    patches = []
    for i,(start, end) in enumerate(intervals):
        if i%2 ==0:
            color = '#82c8a0'
        else:
            color = 'orange'
        r = Rectangle(
            (start, 0),      # 左下角 (x=start, y=0)
            0.0,             # 初始宽度为 0（后续更新）
            1.0,             # 高度=1（配合 xaxis_transform -> 跨满 y 轴）
            transform=ax.get_xaxis_transform(),
            facecolor=color, edgecolor='none', alpha=alpha, zorder=0.2
        )
        ax.add_patch(r)
        patches.append(r)
    return patches

def update_span_patches(patches, intervals, cur_t):
    for r, (start, end) in zip(patches, intervals):
        if cur_t <= start:
            r.set_x(start); r.set_width(0.0)
        elif cur_t >= end:
            r.set_x(start); r.set_width(end - start)
        else:
            r.set_x(start); r.set_width(cur_t - start)

intervals = []
to_ids = t[dto!=2000]
hs_ids = t[dhs!=2000]
for i in range(len(to_ids)):
    intervals.append((to_ids[i],hs_ids[i]))

patches = make_span_patches(ax, intervals)
N = len(t)

lines = [
            Line2D([0], [0], color='darkorange', linewidth=2.5, label='Left Heel'),  # 设置细线
            Line2D([0], [0], color='steelblue', linewidth=2.5, label='Right Heel'),
            Line2D([0], [0], linestyle='None', marker='o',
                markerfacecolor='none', markeredgecolor='red',markeredgewidth=1.8,
                markersize=ms_to, label='Toe-off'),
            Line2D([0], [0], linestyle='None', marker='s',
                markerfacecolor='none', markeredgecolor='black',markeredgewidth=1.8,
                markersize=ms_hs, label='Heel-strike'),
            Patch(facecolor='#82c8a0', alpha=0.5, edgecolor='none', label='Left Swing'),
            Patch(facecolor='orange', alpha=0.5, edgecolor='none', label='Left Swing')]
fig.legend(
        handles=lines,
        loc='upper center',  # 图例位置
        fontsize=16,  # 字体大小
        bbox_to_anchor=(0.5, 1.0),  # 图例位置调整到顶部
        ncol=6  # 设置图例项的列数
    )

N = len(t)
interval_ms = 1000 * 9.3 / N   # 一帧对应的物理时间（毫秒）

idx = {'i': 0}                 # 用可变对象在闭包里保存状态

def step():
    i = idx['i']
    if i >= N:
        timer.stop()           # 播放完停止定时器
        return

    cur_t = t[i]

    # —— 你的原更新逻辑 —— #
    update_span_patches(patches, intervals, cur_t)

    m_to = (dto != 2000) & (t <= cur_t)
    m_hs = (dhs != 2000) & (t <= cur_t)
    to.set_offsets(np.c_[t[m_to],  dto[m_to]] if np.any(m_to) else np.empty((0,2)))
    hs.set_offsets(np.c_[t[m_hs],  dhs[m_hs]] if np.any(m_hs) else np.empty((0,2)))

    line_l.set_data(t[:i+1], dt_l[:i+1])
    line_r.set_data(t[:i+1], dt_r[:i+1])

    fig.canvas.draw_idle()     # 触发重绘（由 GUI 事件循环调度）

    idx['i'] = i + 1

# 创建并启动定时器
timer = fig.canvas.new_timer(interval=interval_ms)
timer.add_callback(step)
timer.start()

plt.show()
