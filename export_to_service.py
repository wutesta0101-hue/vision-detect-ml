# 權重交付 —— 把訓練輸出送進 model-service
# 做四件事：從 runs/ 找出 best.pt、讀出類別清單、複製到 weights/、印出要改的環境變數。
# 刻意不自動修改 vision-detect 的任何檔案 —— 跨 repo 的改動應該由人確認。
# 也刻意不自動比對 /labels：可行性驗證第 ⑥ 項要親手撞一次，撞過才知道自動化該檢查什麼。

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def find_weight(run_name):
    # 從 runs/detect/<run_name>/weights/best.pt 取權重
    # best.pt 是 val 表現最好的那個 epoch，不是最後一個 —— 交付要用這個
    p = ROOT / 'runs' / 'detect' / run_name / 'weights' / 'best.pt'
    if not p.is_file():
        print(f"[FAIL] 權重不存在：{p}")
        return None
    return p


def read_names(pt):
    # 讀出權重內建的類別清單。順序即 class_id，決定了推論結果的語意
    try:
        from ultralytics import YOLO
        names = YOLO(str(pt)).names
        return [names[i] for i in sorted(names)]
    except Exception as e:
        print(f"[WARN] 無法讀取類別清單：{e}")
        return None


def stage(pt, version):
    # 複製到 weights/<version>.pt —— 這裡是進版控的交付物，與 runs/ 的暫存輸出分開
    dst = ROOT / 'weights' / f'{version}.pt'
    dst.parent.mkdir(exist_ok=True)
    shutil.copy2(pt, dst)
    return dst


def deliver(src, target_dir):
    # 複製到 model-service 的權重目錄。target 由參數給，不內建路徑
    d = Path(target_dir)
    if not d.is_dir():
        print(f"[FAIL] 目標資料夾不存在：{d}")
        return None
    dst = d / src.name
    shutil.copy2(src, dst)
    return dst


def instructions(version, names, delivered, mount):
    # 印出人要接手做的事。跨 repo 的改動不自動化，由人確認
    # MODEL_WEIGHTS 必須是容器內的絕對路徑 —— 只寫檔名的話 Ultralytics 會
    # 當成官方預訓練權重的名稱去網路上下載，而不是讀掛載進來的檔案
    filename = delivered.name if delivered else f'{version}.pt'
    print("\n" + "=" * 56)
    print("接下來在 vision-detect 手動執行：\n")
    if delivered:
        print(f"  權重已複製到  {delivered}")
    print(f"  MODEL_WEIGHTS={mount.rstrip('/')}/{filename}   ← 容器內路徑，非 host 路徑")
    print(f"  MODEL_VERSION={version}")
    print("\n  docker compose up -d --force-recreate model")
    if names:
        print(f"\n  預期 /labels 回傳 {len(names)} 個類別：{names}")
    print("=" * 56)


def main():
    ap = argparse.ArgumentParser(description='把訓練權重交付給 model-service')
    ap.add_argument('run_name', help='runs/detect/ 底下的實驗名，如 hardhat-100-v1')
    ap.add_argument('--version', help='MODEL_VERSION，預設與 run_name 相同')
    ap.add_argument('--target', help='model-service 的權重目錄；省略則只做到 weights/')
    ap.add_argument('--mount', default='/app/weights',
                    help='容器內的掛載點，用於組出 MODEL_WEIGHTS 的值')
    args = ap.parse_args()

    version = args.version or args.run_name
    pt = find_weight(args.run_name)
    if not pt:
        return 1

    names = read_names(pt)
    staged = stage(pt, version)
    print(f"已存放  {staged}")

    delivered = None
    if args.target:
        delivered = deliver(staged, args.target)
        if not delivered:
            return 1

    instructions(version, names, delivered, args.mount)
    return 0


if __name__ == '__main__':
    sys.exit(main())
