# 學術圖表配色模組 —— 灰階承載結構，彩色只標示需要立即注意的訊號
# 前五色與 Mermaid 淺色流程圖共用，流程圖與數據圖放在同一份報告裡視覺是連續的

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ══ 共用調色盤 ═══════════════════════════════════════════════
INK  = '#15191E'   # 文字、標題、最強調標記      （= Mermaid color）
MUT  = '#78899B'   # 軸刻度、次要文字、平均線
EDGE = '#B4BAC1'   # 軸線、邊框                  （= Mermaid base stroke）
LT   = '#DFE2E5'   # 參照帶、熱圖中段            （= Mermaid pivot fill）
PALE = '#F4F5F6'   # 最淺填色                    （= Mermaid soft fill）

ACC  = '#1B8FA6'   # 訊號色 · 正常（青）
HOT  = '#C77B14'   # 訊號色 · 超標（琥珀）
THR  = '#C43B45'   # 門檻線（紅）—— 全圖僅此一用途

# 單色→琥珀漸層，供熱圖使用；vmin 應設為「無效應」的基準值
CMAP_HOT = LinearSegmentedColormap.from_list('mono_hot', ['#FFFFFF', LT, HOT])
# 需要雙向（低於/高於基準）時使用
CMAP_DIV = LinearSegmentedColormap.from_list('div', [ACC, '#FFFFFF', HOT])


# 中文字型候選 —— matplotlib 預設的 DejaVu Sans 不含中文，會畫成空白方框。
# 依序嘗試，找不到就退回預設（圖仍可用，只是中文變方框）。
CJK_FONTS = ['Microsoft JhengHei', 'Microsoft YaHei', 'Noto Sans CJK TC',
             'PingFang TC', 'DejaVu Sans']


def apply(base_size=8):
    # 套用全域樣式：中文字型、去掉上右邊框、軸刻度降階為 MUT、圖例不加框
    plt.rcParams.update({
        'font.sans-serif'   : CJK_FONTS,
        'axes.unicode_minus': False,   # 用中文字型時負號會變方框，關掉改用 ASCII 減號
        'font.size'         : base_size,
        'axes.edgecolor'    : EDGE,
        'axes.linewidth'    : 0.8,
        'xtick.color'       : MUT,
        'ytick.color'       : MUT,
        'axes.labelcolor'   : INK,
        'text.color'        : INK,
        'axes.spines.top'   : False,
        'axes.spines.right' : False,
        'figure.facecolor'  : 'white',
        'savefig.facecolor' : 'white',
        'legend.frameon'    : False,
        'axes.titlecolor'   : INK,
    })


def title(ax, main, sub=None, size=9):
    # 主標題短而粗，副說明另起一行降階為 MUT —— 避免兩層資訊擠在同一行
    ax.set_title(main, fontsize=size, weight='bold', loc='left', pad=10 if sub else 4)
    if sub:
        ax.text(0.0, 1.015, sub, transform=ax.transAxes,
                fontsize=size - 1.5, color=MUT, va='bottom')
