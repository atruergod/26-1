import matplotlib.pyplot as plt
import networkx as nx
import platform

import sys as _sys, os as _os
import warnings
import matplotlib as _mpl, matplotlib.font_manager as _fm

warnings.filterwarnings('ignore')

def _setup_korean_font():
    """Windows / macOS / Linux에서 한국어 폰트를 자동 감지해 matplotlib 기본 폰트로 설정."""
    _candidates = {
        'win32': [
            ('C:/Windows/Fonts/malgun.ttf',  'Malgun Gothic'),
            ('C:/Windows/Fonts/gulim.ttc',   'Gulim'),
            ('C:/Windows/Fonts/batang.ttc',  'Batang'),
        ],
        'darwin': [
            ('/System/Library/Fonts/AppleSDGothicNeo.ttc',               'Apple SD Gothic Neo'),
            ('/Library/Fonts/NanumGothic.ttf',                           'NanumGothic'),
            ('/usr/share/fonts/truetype/nanum/NanumGothic.ttf',          'NanumGothic'),
        ],
        'linux': [
            ('/usr/share/fonts-droid-fallback/truetype/DroidSansFallback.ttf', 'Droid Sans Fallback'),
            ('/usr/share/fonts/truetype/nanum/NanumGothic.ttf',                'NanumGothic'),
            ('/usr/share/fonts/truetype/droid/DroidSansFallback.ttf',          'Droid Sans Fallback'),
        ],
    }

    # 깨진 Full 변종 제거 (Linux 한정 이슈)
    _fm.fontManager.ttflist = [
        f for f in _fm.fontManager.ttflist
        if not (f.name == 'Droid Sans Fallback' and 'Full' in f.fname)
    ]

    platform = _sys.platform
    paths = _candidates.get(platform, _candidates['linux'])

    for path, name in paths:
        if _os.path.exists(path):
            _fm.fontManager.addfont(path)
            # 한글 폰트를 맨 앞에 두어 제목/축라벨 한글 깨짐 방지
            _mpl.rcParams['font.family'] = [name, 'DejaVu Sans']
            _mpl.rcParams['font.sans-serif'] = [
                name, 'Malgun Gothic', 'Apple SD Gothic Neo',
                'NanumGothic', 'Droid Sans Fallback', 'DejaVu Sans'
            ]
            print(f"[Font] Using Korean font: {name}")
            return name

    # 한국어 전용 폰트를 못 찾은 경우: 가능한 sans-serif fallback 목록 구성
    _mpl.rcParams['font.family'] = ['DejaVu Sans']
    _mpl.rcParams['font.sans-serif'] = [
        'Malgun Gothic', 'Apple SD Gothic Neo',
        'NanumGothic', 'Droid Sans Fallback', 'DejaVu Sans'
    ]
    print('[Font] Korean font not found explicitly. Using fallback sans-serif chain.')
    return None

_setup_korean_font()
_mpl.rcParams['axes.unicode_minus'] = False


G_sem = nx.DiGraph()

# 노드 정의
latent_nodes = ['X', 'M', 'Y']
control_node = ['Gender']
indicators_x = [f'y{i}' for i in range(1, 5)]
indicators_m = [f'y{i}' for i in range(5, 16)]
indicators_y = [f'y{i}' for i in range(16, 22)]

G_sem.add_nodes_from(latent_nodes + control_node + indicators_x + indicators_m + indicators_y)

# 경로 설정
structural_edges = [('X', 'M'), ('M', 'Y'), ('X', 'Y'), ('Gender', 'M'), ('Gender', 'Y')]
G_sem.add_edges_from(structural_edges)

for i in indicators_x: G_sem.add_edge('X', i)
for i in indicators_m: G_sem.add_edge('M', i)
for i in indicators_y: G_sem.add_edge('Y', i)

# 좌표 설정 (v3: 상단 배치 및 겹침 방지)
pos = {}
pos['X'] = (0, 10)
pos['M'] = (15, 15)
pos['Y'] = (30, 10)
pos['Gender'] = (25, 20)

for idx, i in enumerate(indicators_x): pos[i] = (-6, 7 + idx * 2)
for idx, i in enumerate(indicators_m):
    pos[i] = (5 + (idx % 3) * 2.5, 20 + (idx // 3) * 1.5) 
for idx, i in enumerate(indicators_y): pos[i] = (36, 5 + idx * 2)

plt.figure(figsize=(20, 14))

# 노드 그리기
nx.draw_networkx_nodes(G_sem, pos, nodelist=latent_nodes, node_color='skyblue', node_size=4500)
nx.draw_networkx_nodes(G_sem, pos, nodelist=control_node, node_color='lightgrey', node_size=3500, node_shape='s')
nx.draw_networkx_nodes(G_sem, pos, nodelist=indicators_x + indicators_m + indicators_y, 
                       node_color='white', edgecolors='black', node_size=1000, node_shape='s')

# 경로 그리기
nx.draw_networkx_edges(G_sem, pos, edgelist=structural_edges, width=2.5, arrowsize=25, edge_color='darkblue', connectionstyle='arc3,rad=0.05')
measurement_edges = [e for e in G_sem.edges() if e not in structural_edges]
nx.draw_networkx_edges(G_sem, pos, edgelist=measurement_edges, width=1.0, arrowsize=15, edge_color='grey', style='dashed', alpha=0.4)

# 한글 라벨링
labels = {'X': '쓰기인식\n(Awareness)', 'M': '쓰기반응\n(Reaction)', 'Y': '수행태도\n(Performance)', 'Gender': '성별\n(Gender)'}
nx.draw_networkx_labels(G_sem, pos, labels=labels, font_size=11, font_weight='bold')
nx.draw_networkx_labels(G_sem, pos, labels={n:n for n in indicators_x+indicators_m+indicators_y}, font_size=9)

# 경로 계수 표시
edge_labels = {('X', 'M'): 'β1', ('M', 'Y'): 'β2', ('X', 'Y'): 'γ1', ('Gender', 'M'): 'γM', ('Gender', 'Y'): 'γY'}
nx.draw_networkx_edge_labels(G_sem, pos, edge_labels=edge_labels, font_size=14, label_pos=0.3)

plt.title("한국어 쓰기 태도 구조방정식 모델 (SEM Path Diagram)", fontsize=20)
plt.axis('off')
plt.savefig('sem_path_diagram_korean.png', dpi=300, bbox_inches='tight')
plt.show()