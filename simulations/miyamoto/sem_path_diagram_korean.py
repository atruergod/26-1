"""
sem_path_diagram_korean.py
--------------------------
SEM path diagram for the Korean writing attitude PCM-SEM study.
Layout: M (mediator) raised above X and Y  →  X→Y direct arc
        passes well below M with no overlap.

Run on Windows:
    python sem_path_diagram_korean.py
Output:
    sem_path_diagram_korean.png  (300 dpi, same folder)
"""

import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

# ── Korean font ───────────────────────────────────────────────────────────────
def setup_korean_font():
    for path, name in [
        ('C:/Windows/Fonts/malgun.ttf',  'Malgun Gothic'),
        ('C:/Windows/Fonts/gulim.ttc',   'Gulim'),
        ('/System/Library/Fonts/AppleSDGothicNeo.ttc', 'Apple SD Gothic Neo'),
        ('/Library/Fonts/NanumGothic.ttf', 'NanumGothic'),
        ('/usr/share/fonts/truetype/nanum/NanumGothic.ttf', 'NanumGothic'),
    ]:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            plt.rcParams['font.family'] = name
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            print(f'[font] {name}')
            return
    print('[font] Korean font not found.')

setup_korean_font()
plt.rcParams['axes.unicode_minus'] = False

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

def T(x, y):
    """Pass-through: coordinates already in axes-fraction."""
    return x, y

# ── Key positions  (triangular layout: M above X–Y baseline) ─────────────────
PX = (0.15, 0.50)    # 쓰기인식  (Awareness)
PM = (0.50, 0.72)    # 쓰기반응  (Reaction)   ← raised
PY = (0.85, 0.50)    # 수행태도  (Performance)
PG = (0.50, 0.94)    # 성별     (Gender)

R  = 0.075           # latent circle radius
HB = 0.040           # obs-box half-height
WB = 0.028           # obs-box half-width

# ── Observed indicator positions ──────────────────────────────────────────────
# X indicators: y1–y4, vertical column LEFT of X
IX = [(0.04, PX[1] - 0.09 + i * 0.06) for i in range(4)]

# M indicators: y5–y15, TWO rows BELOW M
#   Row 1 (y5–y10):  6 items centred on PM[0]
#   Row 2 (y11–y15): 5 items centred on PM[0]
IM_Y1 = 0.33
IM_Y2 = 0.20
IM = ([(PM[0] + (k - 2.5) * 0.068, IM_Y1) for k in range(6)] +
      [(PM[0] + (k - 2.0) * 0.068, IM_Y2) for k in range(5)])

# Y indicators: y16–y21, vertical column RIGHT of Y
IY = [(0.96, PY[1] - 0.12 + i * 0.06) for i in range(6)]

# ── Drawing helpers ───────────────────────────────────────────────────────────
def circle(cx, cy, r, fc='#b8d4ea', ec='#1a3a6e', lw=2.0, z=3):
    ax.add_patch(plt.Circle((cx, cy), r, fc=fc, ec=ec, lw=lw, zorder=z,
                             transform=ax.transAxes, clip_on=False))

def rect(cx, cy, hw, hh, fc='white', ec='black', lw=1.2, z=3):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - hw, cy - hh), hw * 2, hh * 2,
        boxstyle='square,pad=0', fc=fc, ec=ec, lw=lw, zorder=z,
        transform=ax.transAxes, clip_on=False))

def label(x, y, s, fs=10, fw='normal', color='black', z=5,
          ha='center', va='center'):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, fontweight=fw,
            color=color, zorder=z, transform=ax.transAxes, clip_on=False)

def sarrow(x0, y0, x1, y1, rad=0.0, color='#1a3a6e', lw=2.2):
    """Structural arrow with solid arrowhead."""
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                xycoords='axes fraction', textcoords='axes fraction',
                zorder=4,
                arrowprops=dict(arrowstyle='-|>', color=color,
                                lw=lw, mutation_scale=22,
                                connectionstyle=f'arc3,rad={rad}'))

def marrow(x0, y0, x1, y1):
    """Measurement arrow (dashed, grey)."""
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                xycoords='axes fraction', textcoords='axes fraction',
                zorder=3,
                arrowprops=dict(arrowstyle='-|>', color='#999999',
                                lw=0.8, mutation_scale=9,
                                linestyle=(0, (5, 3)),
                                connectionstyle='arc3,rad=0.0',
                                alpha=0.65))

def clabel(x, y, txt, color='#1a3a6e', fs=14, dx=0.0, dy=0.022):
    ax.text(x + dx, y + dy, txt, ha='center', va='bottom',
            fontsize=fs, fontstyle='italic', fontweight='bold',
            color=color, zorder=6, transform=ax.transAxes, clip_on=False)

# ── Latent circles ────────────────────────────────────────────────────────────
for (cx, cy), main, sub in [
        (PX, '쓰기인식', '(Awareness)'),
        (PM, '쓰기반응', '(Reaction)'),
        (PY, '수행태도', '(Performance)')]:
    circle(cx, cy, R)
    label(cx, cy + 0.018, main, fs=15, fw='bold')
    label(cx, cy - 0.022, sub,  fs=15, color='#333333')

# ── Gender box ────────────────────────────────────────────────────────────────
rect(PG[0], PG[1], 0.058, 0.034, fc='#e2e2e2', ec='#555555', lw=1.8)
label(PG[0], PG[1] + 0.013, '성별',      fs=14, fw='bold')
label(PG[0], PG[1] - 0.015, '(Gender)',  fs=11, color='#444444')

# ── Observed boxes ────────────────────────────────────────────────────────────
for i, (cx, cy) in enumerate(IX):
    rect(cx, cy, WB, HB); label(cx, cy, f'y{i+1}', fs=15)
for i, (cx, cy) in enumerate(IM):
    rect(cx, cy, WB, HB); label(cx, cy, f'y{i+5}', fs=15)
for i, (cx, cy) in enumerate(IY):
    rect(cx, cy, WB, HB); label(cx, cy, f'y{i+16}', fs=15)

# ── Structural paths ──────────────────────────────────────────────────────────
# β₁: X → M  (diagonal up-right)
A = (PX[0] + R * 0.55, PX[1] + R * 0.84)
B = (PM[0] - R * 0.55, PM[1] - R * 0.84)
sarrow(*A, *B)
clabel((A[0]+B[0])/2 - 0.04, (A[1]+B[1])/2, 'β₁', dy=0.018)

# β₂: M → Y  (diagonal down-right)
A = (PM[0] + R * 0.55, PM[1] - R * 0.84)
B = (PY[0] - R * 0.55, PY[1] + R * 0.84)
sarrow(*A, *B)
clabel((A[0]+B[0])/2 + 0.04, (A[1]+B[1])/2, 'β₂', dy=0.018)

# γ₁: X → Y  (direct, straight line at circle mid-height)
sarrow(PX[0] + R, PX[1],
       PY[0] - R, PY[1],
       rad=0.0)
clabel(0.50, PX[1], 'γ₁ (직접)', dy=-0.052, color='#1a3a6e', fs=13)

# γ_M: Gender → M
sarrow(PG[0] - 0.006, PG[1] - 0.034, PM[0], PM[1] + R,
       rad=0.0, color='#555555', lw=1.8)
clabel(PG[0] - 0.055, (PG[1] + PM[1] + R) / 2, 'γ_M',
       color='#444444', fs=11, dy=0.0)

# γ_Y: Gender → Y  (diagonal right-down from Gender)
sarrow(PG[0] + 0.028, PG[1] - 0.020,
       PY[0] + R * 0.15, PY[1] + R * 0.98,
       rad=-0.08, color='#555555', lw=1.8)
clabel((PG[0] + PY[0]) / 2 + 0.04, (PG[1] + PY[1]) / 2, 'γ_Y',
       color='#444444', fs=11, dy=0.0)

# ── Measurement paths ─────────────────────────────────────────────────────────
for (cx, cy) in IX:
    marrow(PX[0] - R, PX[1], cx + WB, cy)
for (cx, cy) in IM:
    marrow(PM[0], PM[1] - R, cx, cy + HB)
for (cx, cy) in IY:
    marrow(PY[0] + R, PY[1], cx - WB, cy)

# ── Legend & title ────────────────────────────────────────────────────────────
ax.legend(handles=[
    mpatches.Patch(fc='#b8d4ea', ec='#1a3a6e', lw=1.5, label='잠재변수 (원)'),
    mpatches.Patch(fc='white',   ec='black',   lw=1.2, label='관측문항 (사각형)'),
    mpatches.Patch(fc='#e2e2e2', ec='#555555', lw=1.4, label='성별 공변량'),
], loc='lower right', fontsize=12, framealpha=0.9, edgecolor='#cccccc')

ax.set_title('한국어 쓰기 태도 PCM-SEM 경로 모형\n'
             '원: 잠재변수, 사각형: 관측문항(y1–y21), 회색 사각형: 성별 공변량',
             fontsize=16, fontweight='bold', pad=14)

# ── Save ──────────────────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'sem_path_diagram_korean.png')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {OUT}')
