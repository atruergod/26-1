import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_professional_sem():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 스타일 설정
    latent_style = dict(edgecolor='black', facecolor='#e3f2fd', linewidth=1.5)
    manifest_style = dict(edgecolor='black', facecolor='white', linewidth=1)
    
    # 1. 잠재 변수 (Latent Variables) 좌표 설정
    # (x, y) 중심점 및 타원 반지름
    rx, ry = 1.0, 0.7
    pos_x = (2.5, 4.5)
    pos_m = (7.0, 4.5)
    pos_y = (11.5, 4.5)

    # 타원 생성 및 추가
    for pos, label, sub in [(pos_x, r'$\theta_X$', 'Cognitive'), 
                            (pos_m, r'$\theta_M$', 'Affective'), 
                            (pos_y, r'$\theta_Y$', 'Conative')]:
        ellipse = patches.Ellipse(pos, rx*2, ry*2, **latent_style)
        ax.add_patch(ellipse)
        ax.text(pos[0], pos[1], label, fontsize=18, ha='center', va='center')
        ax.text(pos[0], pos[1]+0.9, sub, fontsize=11, ha='center', fontweight='bold')

    # 2. 관측 변수 (Manifest Items) - 타원 상단 배치
    def draw_manifests(center, count):
        width, height = 0.4, 0.4
        gap = 0.1
        total_w = count * width + (count-1) * gap
        start_x = center[0] - total_w/2
        for i in range(count):
            rect_x = start_x + i*(width+gap)
            rect_y = center[1] + 1.8
            rect = patches.Rectangle((rect_x, rect_y), width, height, **manifest_style)
            ax.add_patch(rect)
            # 타원 경계에서 박스 하단으로 연결선
            ax.plot([center[0], rect_x + width/2], [center[1] + ry, rect_y], 
                    color='lightgray', lw=0.8, zorder=0)

    draw_manifests(pos_x, 4)
    draw_manifests(pos_m, 6) # 시각적 명료성을 위해 6개만 표시
    draw_manifests(pos_y, 6)

    # 3. 구조적 경로 (Structural Paths) - 화살표가 타원 경계에 딱 닿도록 조정
    # X -> M
    ax.annotate('', xy=(pos_m[0]-rx, pos_m[1]), xytext=(pos_x[0]+rx, pos_x[1]),
                arrowprops=dict(arrowstyle='-|>', lw=2, color='black', shrinkA=0, shrinkB=0))
    ax.text((pos_x[0]+pos_m[0])/2, pos_m[1]+0.2, r'$\beta_1$', fontsize=15, ha='center')

    # M -> Y
    ax.annotate('', xy=(pos_y[0]-rx, pos_y[1]), xytext=(pos_m[0]+rx, pos_m[1]),
                arrowprops=dict(arrowstyle='-|>', lw=2, color='black', shrinkA=0, shrinkB=0))
    ax.text((pos_m[0]+pos_y[0])/2, pos_y[1]+0.2, r'$\beta_2$', fontsize=15, ha='center')

    # X -> Y (Direct Effect) - 상단 곡선으로 중첩 완전 제거
    # 타원 상단 정점에서 시작하여 상단 정점으로 도착
    arc = patches.FancyArrowPatch((pos_x[0], pos_x[1]+ry), (pos_y[0], pos_y[1]+ry),
                                  connectionstyle="arc3,rad=-0.4", arrowstyle='-|>', 
                                  lw=1.5, ls='--', color='red', mutation_scale=20)
    ax.add_patch(arc)
    ax.text((pos_x[0]+pos_y[0])/2, 7.2, r'$\gamma_1$ (Direct Effect)', fontsize=13, ha='center', color='red')

    # 4. 공변량: 성별 (Gender) - 하단 배치
    gender_pos = (7.0, 1.5)
    gender_box = patches.Rectangle((gender_pos[0]-1.0, gender_pos[1]-0.4), 2.0, 0.8, 
                                   edgecolor='black', facecolor='#eeeeee', linewidth=1.5)
    ax.add_patch(gender_box)
    ax.text(gender_pos[0], gender_pos[1], 'Gender', fontsize=12, ha='center', va='center', fontweight='bold')

    # Gender -> M, Gender -> Y 경로 (타원 하단 경계로 연결)
    # Gender -> M (수직)
    ax.annotate('', xy=(pos_m[0], pos_m[1]-ry), xytext=(gender_pos[0], gender_pos[1]+0.4),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.2, shrinkA=2, shrinkB=2))
    ax.text(7.2, 3.0, r'$\gamma_M$', color='blue', fontsize=13)

    # Gender -> Y (대각선)
    ax.annotate('', xy=(pos_y[0]-0.3, pos_y[1]-ry+0.1), xytext=(gender_pos[0]+1.0, gender_pos[1]),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.2, shrinkA=2, shrinkB=2))
    ax.text(9.5, 2.3, r'$\gamma_Y$', color='blue', fontsize=13)

    # 5. 오차항 (Residuals)
    for pos, label in [(pos_m, r'$\zeta_M$'), (pos_y, r'$\zeta_Y$')]:
        ax.annotate('', xy=(pos[0], pos[1]-ry), xytext=(pos[0], pos[1]-1.3),
                    arrowprops=dict(arrowstyle='<-', lw=1, color='black'))
        ax.text(pos[0]+0.2, pos[1]-1.4, label, fontsize=12)

    plt.tight_layout()
    plt.savefig('standard_sem_final.png', dpi=300, bbox_inches='tight')
    plt.show()

draw_professional_sem()