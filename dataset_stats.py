# 資料集統計 —— 下載後、訓練前的檢查
# 用途：確認 data.yaml 路徑正確、類別分布是否失衡、框大小分布是否偏小物件。
# 這三件事會直接影響評估報告怎麼讀 —— 樣本少的類別 mAP 低是必然，不是模型的問題。
# 輸出：主控台表格 + reports/figures/dataset-classes.png

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

import figstyle

# 類別框數低於總數的這個比例時，在圖上標為「樣本偏少」
# 經驗值，非官方門檻 —— 目的是讓失衡在圖上一眼可見
RARE_RATIO = 0.05


def load_dataset(yaml_path):
    # 讀 data.yaml，回傳 (資料集根目錄, {split: 影像資料夾}, 類別名稱清單)
    cfg = yaml.safe_load(Path(yaml_path).read_text(encoding='utf-8'))
    root = Path(cfg['path']) if 'path' in cfg else Path(yaml_path).parent
    splits = {k: cfg[k] for k in ('train', 'val', 'test') if k in cfg}
    names = cfg['names']
    # names 可能是 list 或 {index: name} 的 dict，統一成 list
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    return root, splits, names


def label_dir(root, images_rel):
    # YOLO 慣例：影像在 .../images，標註在同層的 .../labels
    p = (root / images_rel).resolve()
    return Path(str(p).replace('images', 'labels'))


def box_size(vals):
    # 從一行標註的數值取出框的寬高。支援兩種格式：
    #   偵測格式 4 個值：cx cy w h            → 直接取 w h
    #   分割格式 6+ 偶數個：x1 y1 x2 y2 ...   → 取多邊形的外接框
    # Roboflow 在原始標註是多邊形時會匯出後者，Ultralytics 訓練偵測模型時
    # 會自己轉成外接框，所以統計也用同樣的方式才對得起來
    if len(vals) == 4:
        return vals[2], vals[3]
    if len(vals) >= 6 and len(vals) % 2 == 0:
        xs, ys = vals[0::2], vals[1::2]
        return max(xs) - min(xs), max(ys) - min(ys)
    return None


def scan(dir_path):
    # 掃一個 split 的所有標註檔，回傳 (每類框數, 每張圖框數, 每個框的相對面積, 空標註數)
    counts, per_image, areas, empty = Counter(), [], [], 0
    for f in sorted(dir_path.glob('*.txt')):
        lines = [ln.split() for ln in f.read_text().splitlines() if ln.strip()]
        if not lines:
            empty += 1
        per_image.append(len(lines))
        for parts in lines:
            counts[int(parts[0])] += 1
            wh = box_size([float(v) for v in parts[1:]])
            if wh:
                areas.append(wh[0] * wh[1])
    return counts, per_image, areas, empty


def show(split, names, counts, per_image, areas, empty):
    # 印出單一 split 的統計。框大小用百分位數，比平均值更能看出偏態
    total = sum(counts.values())
    print(f"\n── {split} " + "─" * 44)
    print(f"影像 {len(per_image)} 張 · 框 {total} 個 · 空標註 {empty} 張")
    if not total:
        return
    print(f"每張圖框數  平均 {np.mean(per_image):.1f} · 最多 {max(per_image)}")
    a = np.array(areas)
    print(f"框相對面積  P10 {np.percentile(a,10):.4f} · "
          f"中位 {np.median(a):.4f} · P90 {np.percentile(a,90):.4f}")
    print()
    for i, name in enumerate(names):
        n = counts.get(i, 0)
        flag = "  ← 樣本偏少" if n < total * RARE_RATIO else ""
        print(f"  {i}  {name:<16} {n:>6}  {n/total:6.1%}{flag}")


def plot(names, counts, out_path):
    # 類別分布長條圖。樣本偏少的類別自動變琥珀色 —— 資料自己決定顏色，不人工標註
    import matplotlib.pyplot as plt
    figstyle.apply()
    total = sum(counts.values())
    y = [counts.get(i, 0) for i in range(len(names))]
    colors = [figstyle.HOT if v < total * RARE_RATIO else figstyle.ACC for v in y]

    # 圖寬依類別數縮放，標籤只在名稱長時才旋轉 —— 三類別不該用六類別的版面
    fig, ax = plt.subplots(figsize=(max(3.2, len(names) * 0.9), 3.0))
    ax.bar(range(len(names)), y, color=colors, width=.6)
    ax.set_xticks(range(len(names)))
    rot = 30 if max(len(n) for n in names) > 8 else 0
    ax.set_xticklabels(names, rotation=rot, ha='right' if rot else 'center')
    ax.set_ylabel('框數')
    figstyle.title(ax, '類別分布',
                   f'琥珀 = 佔比低於 {RARE_RATIO:.0%}，該類別 mAP 預期偏低')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\n圖表已存至 {out_path}")


def main():
    ap = argparse.ArgumentParser(description='YOLO 資料集統計')
    ap.add_argument('data_yaml', help='data.yaml 的路徑')
    ap.add_argument('--out', default='reports/figures/dataset-classes.png')
    ap.add_argument('--no-plot', action='store_true', help='只印表格，不畫圖')
    args = ap.parse_args()

    root, splits, names = load_dataset(args.data_yaml)
    print(f"資料集根目錄  {root}")
    print(f"類別 {len(names)} 個  {names}")

    total_counts = Counter()
    for split, rel in splits.items():
        d = label_dir(root, rel)
        if not d.is_dir():
            print(f"\n[FAIL] {split} 的標註資料夾不存在：{d}")
            return 1
        counts, per_image, areas, empty = scan(d)
        show(split, names, counts, per_image, areas, empty)
        total_counts.update(counts)

    if not total_counts:
        print("\n[FAIL] 沒有讀到任何標註框，檢查 data.yaml 的路徑設定。")
        return 1

    if not args.no_plot:
        plot(names, total_counts, Path(args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
