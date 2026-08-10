# 失敗分析 —— 找出模型錯得最離譜的圖，並把錯誤畫出來
# 評估報告給的是「哪個類別差」，這支腳本回答「為什麼差」。
#
# 做法：對每張圖比對預測與標註，計算漏檢(FN)與誤報(FP)數量，
#       取最差的 N 張畫圖。灰色 = 正確，琥珀 = 漏檢，紅 = 誤報。
#       灰階承載結構，彩色只標示需要注意的訊號。

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

import figstyle

ROOT = Path(__file__).parent.resolve()
IOU_THR = 0.45         # 與 Ultralytics ConfusionMatrix 的預設一致，便於與部署表對照
CONF_THR = 0.25        # 與 model-service 的 CONFIDENCE_THRESHOLD 一致


def load_dataset(weights, split):
    # 從權重內建的訓練設定取得 data.yaml，再解析出該 split 的影像資料夾
    from ultralytics import YOLO
    model = YOLO(weights)
    data_path = Path(model.overrides.get('data') or model.ckpt['train_args']['data'])
    cfg = yaml.safe_load(data_path.read_text(encoding='utf-8'))
    root = Path(cfg['path']) if 'path' in cfg else data_path.parent
    names = cfg['names']
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    return model, (root / cfg[split]).resolve(), names


def parse_label(txt_path, w, h):
    # 讀 YOLO 標註檔，回傳 [(類別, x1, y1, x2, y2), ...]，單位為像素
    # 支援兩種格式：偵測 4 值(cx cy w h)、分割 6+ 偶數值(多邊形頂點取外接框)
    if not txt_path.is_file():
        return []
    out = []
    for line in txt_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls, v = int(parts[0]), [float(x) for x in parts[1:]]
        if len(v) == 4:
            cx, cy, bw, bh = v
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        elif len(v) >= 6 and len(v) % 2 == 0:
            xs, ys = v[0::2], v[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            continue
        out.append((cls, x1 * w, y1 * h, x2 * w, y2 * h))
    return out


def iou(a, b):
    # 兩個框的交集除以聯集。a、b 皆為 (x1, y1, x2, y2)
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area = lambda c: (c[2] - c[0]) * (c[3] - c[1])
    return inter / (area(a) + area(b) - inter)


def match(gts, preds):
    # 貪婪配對：每個預測找 IoU 最高且類別相同的未配對標註
    # 回傳 (配對成功的預測索引, 漏檢的標註索引, 誤報的預測索引)
    used, ok, fp = set(), [], []
    for pi, p in enumerate(preds):
        best, best_gi = IOU_THR, None
        for gi, g in enumerate(gts):
            if gi in used or g[0] != p[0]:
                continue
            v = iou(g[1:], p[1:])
            if v >= best:
                best, best_gi = v, gi
        if best_gi is None:
            fp.append(pi)
        else:
            used.add(best_gi)
            ok.append(pi)
    fn = [gi for gi in range(len(gts)) if gi not in used]
    return ok, fn, fp


def draw(img_path, gts, preds, ok, fn, fp, names, out_path):
    # 把三種框畫在同一張圖上。正確的用灰色弱化，錯誤的才上色
    from PIL import Image, ImageDraw
    im = Image.open(img_path).convert('RGB')
    d = ImageDraw.Draw(im)
    for pi in ok:
        box(d, preds[pi], names, figstyle.EDGE, 2)
    for gi in fn:
        box(d, gts[gi], names, figstyle.HOT, 3, '漏')
    for pi in fp:
        box(d, preds[pi], names, figstyle.THR, 3, '誤')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def box(d, item, names, color, width, tag=''):
    # 畫單一個框與標籤。標籤用英文類別名，避免字型問題
    cls, x1, y1, x2, y2 = item
    d.rectangle([x1, y1, x2, y2], outline=color, width=width)
    label = f"{tag}{names[cls]}" if tag else names[cls]
    d.text((x1 + 3, max(0, y1 - 12)), label, fill=color)


def summarise(rows, names):
    # 統計每個類別的漏檢與誤報總數 —— 看出問題集中在哪一類
    fn_c, fp_c = Counter(), Counter()
    for r in rows:
        fn_c.update(r['fn_cls'])
        fp_c.update(r['fp_cls'])
    print(f"\n{'類別':<16}{'漏檢':>8}{'誤報':>8}")
    print("─" * 32)
    for i, n in enumerate(names):
        print(f"{n:<16}{fn_c[i]:>8}{fp_c[i]:>8}")
    print("─" * 32)
    print(f"{'合計':<16}{sum(fn_c.values()):>8}{sum(fp_c.values()):>8}")


def main():
    ap = argparse.ArgumentParser(description='YOLO 失敗分析')
    ap.add_argument('weights')
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--top', type=int, default=20, help='畫出最差的幾張')
    ap.add_argument('--cls', help='只看某個類別的錯誤，如 Mask')
    ap.add_argument('--out', default='reports/figures/errors')
    args = ap.parse_args()

    if not Path(args.weights).is_file():
        print(f"[FAIL] 權重不存在：{args.weights}")
        return 1

    from PIL import Image
    model, img_dir, names = load_dataset(args.weights, args.split)
    label_dir = Path(str(img_dir).replace('images', 'labels'))
    focus = names.index(args.cls) if args.cls else None
    print(f"影像資料夾  {img_dir}")

    rows = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        w, h = Image.open(img_path).size
        gts = parse_label(label_dir / f'{img_path.stem}.txt', w, h)
        r = model.predict(img_path, conf=CONF_THR, verbose=False)[0]
        preds = [(int(c), *map(float, xy)) for c, xy in
                 zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]

        ok, fn, fp = match(gts, preds)
        fn_cls = [gts[i][0] for i in fn]
        fp_cls = [preds[i][0] for i in fp]
        # 指定類別時，只計該類別的錯誤來排名，其餘照畫但不影響排序
        score = (fn_cls + fp_cls).count(focus) if focus is not None \
            else len(fn) + len(fp)
        rows.append({'path': img_path, 'gts': gts, 'preds': preds, 'ok': ok,
                     'fn': fn, 'fp': fp, 'fn_cls': fn_cls, 'fp_cls': fp_cls,
                     'score': score})

    if not rows:
        print("[FAIL] 沒有讀到任何影像")
        return 1

    summarise(rows, names)

    worst = sorted(rows, key=lambda r: -r['score'])[:args.top]
    out = Path(args.out)
    print(f"\n最差 {len(worst)} 張：")
    for i, r in enumerate(worst, 1):
        if r['score'] == 0:
            break
        p = out / f"{i:02d}_{r['path'].stem}.jpg"
        draw(r['path'], r['gts'], r['preds'], r['ok'], r['fn'], r['fp'], names, p)
        print(f"  {i:2d}  漏檢 {len(r['fn'])} · 誤報 {len(r['fp'])}  {r['path'].name}")
    print(f"\n已存至 {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
