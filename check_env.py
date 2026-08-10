# 環境檢查 —— 可行性驗證第 ① 項
# 用途：套件裝完之後，確認 CUDA 真的可用、VRAM 足夠、版本相容。
# 不檢查「有沒有裝」（沒裝的話 import 就會失敗），只檢查「裝對了沒有」。
# 任一項失敗回傳 exit code 1，可當作後續步驟的前置閘門。

import sys

# 判定門檻：低於此值，yolov8s 在 640x640 下的 batch size 會被迫壓很小
MIN_VRAM_GB = 6.0

# 收集結果，最後統一判定。用 (是否通過, 顯示文字) 的形式
results = []


def record(ok, label, detail):
    # 記錄一項檢查結果並即時印出，讓使用者看到卡在哪一項
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label:16} {detail}")
    results.append(ok)


def check_python():
    # Python 版本 —— Ultralytics 對 3.8 以下不再支援
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 9)
    record(ok, "Python", f"{v.major}.{v.minor}.{v.micro}")


def check_torch(torch):
    # torch 是否為 CUDA 版。CPU 版的 torch.version.cuda 會是 None
    built = torch.version.cuda
    ok = built is not None
    detail = f"{torch.__version__} (built with CUDA {built})" if ok \
        else f"{torch.__version__} —— 這是 CPU 版，需重裝 CUDA 版"
    record(ok, "torch", detail)


def check_cuda(torch):
    # 執行期能否真的看到 GPU。torch 是 CUDA 版但驅動太舊時，這裡會是 False
    ok = torch.cuda.is_available()
    detail = "可用" if ok else "不可用 —— 檢查顯示卡驅動（nvidia-smi）"
    record(ok, "CUDA", detail)
    return ok


def check_gpu(torch):
    # GPU 型號與 VRAM。VRAM 決定 batch size 上限
    p = torch.cuda.get_device_properties(0)
    gb = p.total_memory / 1024 ** 3
    ok = gb >= MIN_VRAM_GB
    record(ok, "GPU", f"{p.name} · {gb:.1f} GB · sm_{p.major}{p.minor}")


def check_ultralytics():
    # Ultralytics 是否可載入。版本本身不設門檻，只記錄下來供日後比對
    try:
        import ultralytics
        record(True, "ultralytics", ultralytics.__version__)
    except ImportError as e:
        record(False, "ultralytics", f"無法載入 —— {e}")


def main():
    print("=" * 56)
    check_python()

    try:
        import torch
    except ImportError as e:
        record(False, "torch", f"無法載入 —— {e}")
        return finish()

    check_torch(torch)
    if check_cuda(torch):
        check_gpu(torch)
    else:
        record(False, "GPU", "略過 —— CUDA 不可用")

    check_ultralytics()
    return finish()


def finish():
    # 統一判定並回傳 exit code，方便串進其他腳本或 CI
    print("=" * 56)
    ok = all(results)
    print("環境檢查通過。" if ok else "環境檢查未通過，請先處理上方 FAIL 項目。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
