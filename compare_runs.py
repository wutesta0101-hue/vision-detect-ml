# 跨實驗比較 —— 把多次訓練畫在同一張圖上
# 回答一個問題：這兩次實驗差在哪。
#
# 資料來源兩處：
#   runs/detect/<name>/results.csv     訓練曲線與成本（Ultralytics 產生）
#   reports/<name>-<split>.md          分類別 FN（evaluate.py 產生，解析其表格）
#
# 第二項是與 evaluate.py 的耦合 —— 那邊的表格格式若改，這裡要跟著改。
# 對照實驗的判準用 FN 而非 FP：資料集存在系統性漏標，FP 與 precision 因此
# 被低估，但 FN（標註有、模型沒抓到）不受影響。

import argparse
import csv
import re
import sys
from pathlib import Path

import figstyle

ROOT = Path(__file__).parent.resolve()
# 依序取用的顏色。第一個實驗用灰（基準），其後用訊號色
SERIES = [figstyle.MUT, figstyle.HOT, figstyle.ACC, figstyle.INK]


def read_curve(name):
    # 讀 results.csv，回傳 (epoch 列表, mAP50 列表, 總秒數)
    p = ROOT / 'runs' / 'detect' / name / 'results.csv'
    if not p.is_file():
        print(f"[WARN] 找不到 {p}")
        return None
    ep, m, sec = [], [], 0.0
    with p.open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            r = {k.strip(): v for k, v in row.items()}
            ep.append(int(float(r['epoch'])))
            m.append(float(r['metrics/mAP50(B)']))
            sec = float(r['time'])
    return ep, m, sec


def read_deploy(name, split):
    # 解析 evaluate.py 產生的「部署閾值下的表現」表格，回傳 {類別: FN}
    # 格式：| 類別 | TP | FP | FN | P | R |
    p = ROOT / 'reports' / f'{name}-{split}.md'
    if not p.is_file():
        print(f"[WARN] 找不到 {p} —— 請先跑 evaluate.py")
        return None
    text = p.read_text(encoding='utf-8')
    if '部署閾值' not in text:
        print(f"[WARN] {p} 沒有部署閾值表格 —— 該次評估可能用了 --conf 0")
        return None
    body = text.split('部署閾值')[1]
    out = {}
    for line in body.splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        # 只取「六欄、第 2-4 欄是數字、類別名不含 ** 」的資料列
        if len(cells) == 6 and '**' not in cells[0] and cells[3].isdigit():
            out[cells[0]] = int(cells[3])
    return out or None


def plot_curves(runs, out_path):
    # 訓練曲線疊圖 —— 看收斂速度與最終水準的差異
    import matplotlib.pyplot as plt
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for i, (name, (ep, m, sec)) in enumerate(runs.items()):
        ax.plot(ep, m, color=SERIES[i % len(SERIES)], lw=1.4,
                label=f'{name}  ({sec/60:.0f} 分)')
    ax.set_xlabel('epoch')
    ax.set_ylabel('val mAP@50')
    ax.legend(loc='lower right')
    figstyle.title(ax, '訓練曲線比較', '括號為總訓練時長')
    save(fig, out_path)


def plot_fn(deploys, out_path, conf_note):
    # 分類別漏檢數並排長條 —— 對照實驗的主要判準
    import matplotlib.pyplot as plt
    import numpy as np
    figstyle.apply()
    names = list(next(iter(deploys.values())).keys())
    x = np.arange(len(names))
    w = 0.8 / len(deploys)

    fig, ax = plt.subplots(figsize=(max(4.5, len(names) * 1.1), 3.2))
    for i, (run, d) in enumerate(deploys.items()):
        ax.bar(x + i * w - 0.4 + w / 2, [d.get(n, 0) for n in names],
               width=w * .9, color=SERIES[i % len(SERIES)], label=run)
    ax.set_xticks(x, names, rotation=30, ha='right')
    ax.set_ylabel('漏檢數 (FN)')
    ax.legend()
    figstyle.title(ax, '分類別漏檢數比較', conf_note)
    save(fig, out_path)


def save(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"圖表已存至 {out_path}")


def print_cost(runs):
    # 成本表。每 epoch 秒數才是可比的 —— 總時長受 early stopping 影響
    print(f"\n{'實驗':<20}{'epochs':>8}{'總時長':>10}{'每 epoch':>11}{'最終 mAP50':>12}")
    print("─" * 61)
    for name, (ep, m, sec) in runs.items():
        n = max(ep)
        print(f"{name:<20}{n:>8}{sec/60:>9.1f}分{sec/n:>10.1f}秒{m[-1]:>12.3f}")


def print_fn(deploys):
    # 漏檢數對照表，含與第一個實驗的差值
    base_name = next(iter(deploys))
    base = deploys[base_name]
    others = [k for k in deploys if k != base_name]
    print(f"\n{'類別':<16}" + "".join(f"{k[:14]:>16}" for k in deploys) +
          ("      差值" if len(others) == 1 else ""))
    print("─" * (16 + 16 * len(deploys) + 10))
    for n in base:
        line = f"{n:<16}" + "".join(f"{d.get(n, 0):>16}" for d in deploys.values())
        if len(others) == 1:
            diff = deploys[others[0]].get(n, 0) - base[n]
            line += f"{diff:>+10}"
        print(line)
    print("─" * (16 + 16 * len(deploys) + 10))
    line = f"{'合計':<16}" + "".join(f"{sum(d.values()):>16}" for d in deploys.values())
    if len(others) == 1:
        line += f"{sum(deploys[others[0]].values()) - sum(base.values()):>+10}"
    print(line)


def main():
    ap = argparse.ArgumentParser(description='跨實驗比較')
    ap.add_argument('names', nargs='+', help='實驗名，第一個為基準')
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--out', default='reports/figures')
    args = ap.parse_args()

    runs = {n: c for n in args.names if (c := read_curve(n))}
    if len(runs) < 2:
        print("[FAIL] 至少要有兩次可讀的實驗")
        return 1
    print_cost(runs)

    deploys = {n: d for n in runs if (d := read_deploy(n, args.split))}
    out = Path(args.out)
    tag = '_vs_'.join(args.names)
    plot_curves(runs, out / f'compare-{tag}-curve.png')

    if len(deploys) >= 2:
        print_fn(deploys)
        plot_fn(deploys, out / f'compare-{tag}-fn.png',
                f'{args.split} 集，conf=0.25；漏檢不受標註漏標干擾')
    else:
        print("\n[WARN] 可讀的部署表格不足兩份，略過漏檢比較圖")
    return 0


if __name__ == '__main__':
    sys.exit(main())
