# 模型評估 —— 分類別 mAP、混淆矩陣、可貼進報告的表格
# Ultralytics 訓練時已產生 PR 曲線與混淆矩陣（在 runs/ 底下），這裡不重畫。
# 這支腳本補三件它沒做好的事：
#   ① 分類別數值只印在 console，訓練結束就沒了 → 寫成 markdown 表格
#   ② 訓練只跑 val，test split 要另外呼叫
#   ③ 配色與流程圖、資料集統計圖一致，報告裡視覺連續
#
# ⚠️ 沒有樣本的類別，Ultralytics 會把它從結果陣列中整個略過，不是給 NaN。
#    所以類別對應必須透過 ap_class_index，不能用陣列位置。

import argparse
import sys
from pathlib import Path

import numpy as np

import figstyle

ROOT = Path(__file__).parent.resolve()


def evaluate(weights, split):
    # 跑一次驗證。data 從權重內建的訓練設定取得，不需另外指定
    # plots=True 是必要的 —— plots=False 時 Ultralytics 不會填充 confusion_matrix，
    # 拿到的會是全 0 的矩陣。副作用是它也會在 runs/ 產生自己的圖，可當交叉比對用。
    from ultralytics import YOLO
    model = YOLO(weights)
    return model.val(split=split, plots=True, verbose=False)


def per_class_rows(res):
    # 把結果攤成每類別一列。ap_class_index 是「有樣本的類別」的索引清單，
    # 沒出現在裡面的類別代表該 split 完全沒有標註框
    names = res.names
    present = {int(c): i for i, c in enumerate(res.box.ap_class_index)}
    rows = []
    for cid in sorted(names):
        if cid in present:
            i = present[cid]
            rows.append({
                'id': cid, 'name': names[cid], 'has_data': True,
                'p': res.box.p[i], 'r': res.box.r[i],
                'map50': res.box.ap50[i], 'map': res.box.maps[cid],
            })
        else:
            rows.append({'id': cid, 'name': names[cid], 'has_data': False})
    return rows


def print_table(rows, res):
    # 主控台輸出。沒有樣本的類別明確標示，避免被誤讀為「模型完全學不會」
    print(f"\n{'類別':<16}{'P':>8}{'R':>8}{'mAP50':>9}{'mAP50-95':>10}")
    print("─" * 51)
    for r in rows:
        if r['has_data']:
            print(f"{r['name']:<16}{r['p']:>8.3f}{r['r']:>8.3f}"
                  f"{r['map50']:>9.3f}{r['map']:>10.3f}")
        else:
            print(f"{r['name']:<16}{'—':>8}{'—':>8}{'—':>9}{'—':>10}   ← 此 split 無樣本")
    print("─" * 51)
    print(f"{'整體':<16}{res.box.mp:>8.3f}{res.box.mr:>8.3f}"
          f"{res.box.map50:>9.3f}{res.box.map:>10.3f}")


def plot_map(rows, out_path):
    # 分類別 mAP@50 長條圖。低於整體平均的類別自動變琥珀 —— 資料自己決定顏色
    import matplotlib.pyplot as plt
    figstyle.apply()
    valid = [r for r in rows if r['has_data']]
    if not valid:
        return
    y = [r['map50'] for r in valid]
    labels = [r['name'] for r in valid]
    mean = float(np.mean(y))
    colors = [figstyle.HOT if v < mean else figstyle.ACC for v in y]

    fig, ax = plt.subplots(figsize=(max(3.2, len(valid) * 0.9), 3.0))
    ax.bar(range(len(valid)), y, color=colors, width=.6)
    ax.axhline(mean, color=figstyle.MUT, ls=':', lw=.9)
    ax.set_xticks(range(len(valid)))
    rot = 30 if max(len(n) for n in labels) > 8 else 0
    ax.set_xticklabels(labels, rotation=rot, ha='right' if rot else 'center')
    ax.set_ylabel('mAP@50')
    ax.set_ylim(0, 1)
    figstyle.title(ax, '分類別 mAP@50',
                   f'虛線 = 整體平均 {mean:.3f}；琥珀 = 低於平均')
    save(fig, out_path)


def plot_confusion(res, names, out_path):
    # 混淆矩陣。用自己的配色重畫，與報告其他圖表一致
    # 矩陣是 (nc+1, nc+1)，最後一列/欄是背景（漏檢與誤報）
    import matplotlib.pyplot as plt
    figstyle.apply()
    m = np.asarray(res.confusion_matrix.matrix, dtype=float)
    if m.sum() == 0:
        print("[WARN] 混淆矩陣為空，略過繪圖 —— 請確認 val() 有帶 plots=True")
        return
    labels = [names[i] for i in sorted(names)] + ['背景']

    fig, ax = plt.subplots(figsize=(max(3.6, len(labels) * 0.85),
                                    max(3.2, len(labels) * 0.75)))
    im = ax.imshow(m, cmap=figstyle.CMAP_HOT, vmin=0)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha='right')
    ax.set_yticks(range(len(labels)), labels)
    # Ultralytics 的矩陣是 [預測][實際] —— 列是預測、欄是實際，與直覺相反
    ax.set_xlabel('實際')
    ax.set_ylabel('預測')

    # 數字的對比度跟著底色切換，深底用白字
    hi = m.max() * 0.6 if m.max() else 1
    for a in range(m.shape[0]):
        for b in range(m.shape[1]):
            v = int(m[a, b])
            if v:
                ax.text(b, a, v, ha='center', va='center', fontsize=7,
                        color='white' if m[a, b] > hi else figstyle.INK)
    figstyle.title(ax, '混淆矩陣', '列 = 預測，欄 = 實際；最後一列/欄為背景')
    save(fig, out_path)


def save(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"圖表已存至 {out_path}")


def write_markdown(rows, res, split, weights, out_path):
    # 產生可直接貼進報告的表格。數值留在檔案裡，不會隨 console 清空而消失
    lines = [f"# 評估結果 · {split}", "",
             f"權重：`{weights}`", "",
             "| 類別 | P | R | mAP@50 | mAP@50-95 |", "|---|---|---|---|---|"]
    for r in rows:
        if r['has_data']:
            lines.append(f"| {r['name']} | {r['p']:.3f} | {r['r']:.3f} | "
                         f"{r['map50']:.3f} | {r['map']:.3f} |")
        else:
            lines.append(f"| {r['name']} | — | — | — | — |  ")
    lines.append(f"| **整體** | **{res.box.mp:.3f}** | **{res.box.mr:.3f}** | "
                 f"**{res.box.map50:.3f}** | **{res.box.map:.3f}** |")

    missing = [r['name'] for r in rows if not r['has_data']]
    if missing:
        lines += ["", f"> `{'`、`'.join(missing)}` 在此 split 沒有標註框，"
                      "無法計算指標 —— 是資料的限制，不是模型的問題。"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding='utf-8')
    print(f"表格已存至 {out_path}")


def main():
    ap = argparse.ArgumentParser(description='YOLO 模型評估')
    ap.add_argument('weights', help='權重路徑，如 weights/hardhat-100-v1.pt')
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--out', default='reports', help='輸出目錄')
    args = ap.parse_args()

    if not Path(args.weights).is_file():
        print(f"[FAIL] 權重不存在：{args.weights}")
        return 1

    res = evaluate(args.weights, args.split)
    rows = per_class_rows(res)
    print_table(rows, res)

    out = Path(args.out)
    stem = Path(args.weights).stem
    plot_map(rows, out / 'figures' / f'{stem}-{args.split}-map.png')
    plot_confusion(res, res.names, out / 'figures' / f'{stem}-{args.split}-confusion.png')
    write_markdown(rows, res, args.split, args.weights,
                   out / f'{stem}-{args.split}.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
