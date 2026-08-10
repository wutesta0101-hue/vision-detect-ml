# 訓練入口 —— 讀 configs/*.yaml，其餘交給 Ultralytics
# 這支腳本刻意很薄：訓練邏輯在 Ultralytics 裡，這裡只負責
#   ① 把設定集中在版本控管的 YAML，而不是散在指令列參數
#   ② 把 data 轉成絕對路徑，避開 Ultralytics 的 datasets_dir 解析
#   ③ 訓練後印出權重位置與下一步指令
# 想調超參數改 configs/*.yaml，不要改這裡。

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.resolve()


def load_config(path):
    # 讀訓練設定。model 是基底權重，其餘全部原樣傳給 Ultralytics 的 train()
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    model = cfg.pop('model')

    # data 轉絕對路徑 —— Ultralytics 解析相對路徑時會先看 settings.json 的
    # datasets_dir，不是工作目錄。轉絕對路徑可完全避開這個行為。
    data = Path(cfg['data'])
    cfg['data'] = str(data if data.is_absolute() else (ROOT / data).resolve())

    # 輸出固定落在 repo 的 runs/，不受 Ultralytics 全域設定影響
    cfg.setdefault('project', str(ROOT / 'runs' / 'detect'))
    return model, cfg


def check_paths(cfg):
    # 訓練前先確認 data.yaml 存在。Ultralytics 的錯誤訊息不會說它找了哪裡
    p = Path(cfg['data'])
    if not p.is_file():
        print(f"[FAIL] data.yaml 不存在：{p}")
        return False
    return True


def show(model, cfg):
    # 把實際生效的設定攤開來印，避免「以為改了但沒生效」
    print("=" * 56)
    print(f"  {'model':8} {model}")
    for k, v in cfg.items():
        print(f"  {k:8} {v}")
    print("=" * 56)


def report(results, cfg):
    # 印出權重位置與下一步。best.pt 是 val 表現最好的那個 epoch，不是最後一個
    out = Path(results.save_dir)
    print(f"\n訓練輸出  {out}")
    print(f"最佳權重  {out / 'weights' / 'best.pt'}")
    print(f"\n下一步：")
    print(f"  .\\.venv\\Scripts\\python.exe evaluate.py {out / 'weights' / 'best.pt'}")


def main():
    ap = argparse.ArgumentParser(description='YOLO fine-tune')
    ap.add_argument('config', help='configs/*.yaml 的路徑')
    ap.add_argument('--dry-run', action='store_true',
                    help='只印設定與路徑檢查，不實際訓練')
    args = ap.parse_args()

    model_name, cfg = load_config(args.config)
    show(model_name, cfg)
    if not check_paths(cfg):
        return 1
    if args.dry_run:
        print("dry-run：設定與路徑正常，未執行訓練。")
        return 0

    from ultralytics import YOLO  # import 較慢，dry-run 時不需要
    results = YOLO(model_name).train(**cfg)
    report(results, cfg)
    return 0


if __name__ == '__main__':
    sys.exit(main())
