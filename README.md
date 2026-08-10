# vision-detect-ml

> YOLO 模型訓練與評估 —— 資料集、fine-tune、評估報告。

產出的權重部署到 [vision-detect](https://github.com/wutesta0101-hue/vision-detect) 的推論服務，  
介面是**一個 `.pt` 檔加兩個環境變數**。本 repo 不依賴該專案的程式碼。

用 Ultralytics YOLOv8s 在 2,114 張 PPE 資料集上做 fine-tune，**自行實作評估與失敗分析工具**。  
逐框查證發現部分 `Mask` 誤報來自標註漏標，而非單純模型能力問題——  
詳見[評估報告](reports/ppe-2114-評估報告.md)。


---

## 專案目錄

- [專案目的](#專案目的)
- [與 vision-detect 的關係](#與-vision-detect-的關係)
- [期望學習的技能](#期望學習的技能)
- [實作順序](#實作順序)
- [訓練流程](#訓練流程)
- [環境建置](#環境建置)
- [資料集](#資料集)
- [評估報告](#評估報告)
- [技術組成](#技術組成)
- [專案結構](#專案結構)
- [已知限制](#已知限制)
- [授權](#授權)

---

## 專案目的

第一階段用 COCO 預訓練模型把系統跑通了。這個 repo 是下一步：fine-tune 一個領域模型，並且能說清楚它好在哪、壞在哪。

多數 YOLO 專案的 README 只寫「用了 YOLO，效果不錯」。這裡要練的是後面那半——分類別 mAP、混淆矩陣、部署閾值下的實際計數、逐框失敗分析，讓專案可以用數字說話。

**訓練以懂評估為目的。**

---

## 與 vision-detect 的關係

![交付邊界](docs/delivery-boundary(zh).png)


兩個 repo 之間只有一條線：訓練完的權重複製過去，改兩個環境變數，重啟一個容器。

```
weights/best.pt
    ↓ 複製
vision-detect/model-service/weights/
修改 MODEL_WEIGHTS 與 MODEL_VERSION
docker compose up -d --force-recreate model
    ↓
完成。C#、Vue、MAUI 完全不動。
```

`MODEL_VERSION` 命名為 `<資料集>-<規模>-v<序號>`（如 `ppe-2114-v1`），與 `runs/` 的實驗名、`weights/` 的檔名一致。

之所以能這樣，是第一階段的架構決定：類別清單由 `/labels` 端點提供、每筆紀錄存 `ModelVersion`、模型參數在設定檔、推論服務是獨立容器。

**訓練不需要專案系統跑；驗證需要。** 見[實作順序](#實作順序)第 ① 步。

---

## 期望學習的技能

| 面向 | 內容 |
|---|---|
| **資料集品質** | 授權確認、標註抽查、類別分布統計——ML 中最反直覺的一課：資料品質決定模型品質 |
| **fine-tune** | Ultralytics 高階封裝下的訓練設定、超參數、GPU 資源配置 |
| **評估分析** | 為什麼「準確率」在偵測任務裡沒有意義；mAP@50 與 mAP@50-95 的差別 |
| **失敗分析** | 從最差的樣本回推問題——多數改進來自修正資料，而非調整超參數 |
| **看懂訓練曲線** | loss 正常長什麼樣、過擬合的徵兆、learning rate 的表現 |

---

## 實作順序

| 順序 | 主題 | 估計 | 狀態 |
|---|---|---|---|
| ① | 可行性驗證（100 張，走完整輪） | 2–4 小時 | 完成 |
| ② | fine-tune 練習（2,114 張 PPE） | 15–25 小時 | 完成 |
| ③ | 評估報告 | 15–20 小時 | 完成 |
| ④ | PyTorch 基礎（觸發式，讀懂訓練曲線） | 3–5 小時 | |
| ⑤ | 大規模資料處理 | 另行規劃 | |

### ① 可行性驗證

用 100 張的小資料集走完整輪。**目的不是得到好模型**，是確認環境與流程沒問題。

- [x] CUDA 環境正常（`torch.cuda.is_available()` 為 True）
- [x] Roboflow 匯出的 YOLO 格式能被 Ultralytics 直接讀取
- [x] `data.yaml` 的路徑設定正確
- [x] 訓練能在 GPU 上跑起來，VRAM 沒爆
- [x] 匯出的權重能塞進 `model-service` 容器
- [x] 換模型後 API 回傳的 `model_version` 與 `/labels` 正確更新
- [x] 桌機儀表板的類別篩選顯示新的類別清單

**後三項需要 vision-detect**（`docker compose up -d`）

與第一階段「先用 COCO 打通鏈路」是同一個邏輯：**先讓最不確定的部分變確定。**

第 ⑤ 項在驗證時發現 vision-detect 的 `docker-compose.yml` 缺少 volume 掛載與 `MODEL_WEIGHTS` 環境變數，「換模型不需重建映像」原本不成立——已修補，過程見[可行性驗證報告](reports/smoke-report.md)。

---

## 訓練流程

![訓練流程](docs/training-flow(zh).png)


虛線是回頭的路。**ML 流程不是直線**——評估發現某類別 mAP 低就回去調參，失敗分析發現標註品質差就回去修資料。

| 腳本 | 用途 |
|---|---|
| `check_env.py` | CUDA / VRAM / 版本檢查 |
| `dataset_stats.py` | 類別分布、每圖物件數、框大小 |
| `train.py` | fine-tune |
| `evaluate.py` | mAP、部署閾值計數、混淆矩陣 |
| `error_analysis.py` | 最差 N 張的錯誤標記與逐框查證 |
| `compare_runs.py` | 跨實驗對照（訓練曲線、分類別漏檢數） |
| `export_to_service.py` | 挑權重、印出要改的環境變數 |
| `figstyle.py` | 圖表配色（與 Mermaid 流程圖共用前五色） |

---

## 環境建置

| 項目 | 內容 |
|---|---|
| OS | Windows 11 / PowerShell 5.1 |
| GPU | RTX 3080 12 GB |
| Python | 3.12 |

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# torch 必須先裝，且走 CUDA index
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_env.py
```

Roboflow 匯出的 `data.yaml` 用相對路徑（`../train/images`），從 repo 根目錄執行會解析錯誤。  
改為絕對路徑：

```yaml
path: C:/dev/PR_vision-detect-ml/vision-detect-ml/datasets/ppe-2114
train: train/images
val: valid/images
test: test/images
```

正斜線——反斜線在 YAML 裡會被當跳脫字元。

---

## 資料集

以 PPE / 工地安全裝備為練習領域。類別少、範圍窄，混淆分析有素材。

| 資料集 | 規模 | 類別 | 用途 |
|---|---|---|---|
| Hard Hat Sample | 100 張 | 3 類 | 可行性驗證 |
| PPE Detection | 2,114 張 | 6 類 | 正式練習 |

每個資料集的來源與授權記在 [`docs/SOURCES.md`](docs/SOURCES.md)。**下載當下就紀錄**

> `datasets/` 與 `runs/` 不進版控。

---

## 評估報告

**[PPE 偵測模型評估報告](reports/ppe-2114-評估報告.md)** —— test mAP@50 0.866，  
含資料集分析、失敗案例逐框查證、以及一次解析度對照實驗（假設被證偽）。

| 產出 | 說明 |
|---|---|
| 資料集統計 | 類別分布、每張圖平均物件數、框大小分布 |
| mAP@50 / mAP@50-95 | 整體與各類別 |
| 部署閾值下的 TP/FP/FN | conf=0.25，與 `model-service` 設定一致，可跨實驗比較 |
| 混淆矩陣 | 模型把什麼誤認成什麼 |
| 失敗案例分析 | 產出最差 20 張；報告逐框查證其中 2 張 |
| 對照實驗 | imgsz 640 vs 960，含證偽條件與訓練成本 |
| 換模型前後對照 | 附上實際的操作成本 |

其他報告：[可行性驗證](reports/smoke-report.md)（100 張走完整輪，發現並修補 vision-detect 的換模型缺口）。

---

## 技術組成

| 用途 | 技術 |
|---|---|
| 訓練 | Ultralytics YOLOv8（PyTorch CUDA） |
| 評估 | 自行實作（numpy · matplotlib） |
| 資料集 | Roboflow Universe |

> 原規劃列有 scikit-learn，實作後未使用——偵測任務的 IoU 配對與 mAP 不在其涵蓋範圍，  
> 而 Ultralytics 已提供官方定義的指標。剩餘的統計用 numpy 即可。

---

## 專案結構

```
vision-detect-ml/
├── check_env.py          環境檢查
├── dataset_stats.py      資料集統計
├── train.py              fine-tune
├── evaluate.py           mAP · 部署閾值 · 混淆矩陣
├── error_analysis.py     失敗分析
├── compare_runs.py       跨實驗對照
├── export_to_service.py  權重交付
├── figstyle.py           圖表配色
├── configs/              訓練設定 YAML
├── datasets/             資料集（不進版控）
├── runs/                 訓練輸出（不進版控）
├── weights/              挑選後的權重
├── reports/              評估報告與圖表
└── docs/                 規劃與圖表
```

`runs/` 是每次訓練的暫存輸出，`weights/` 是**刻意挑出來**要交付的那一個。兩者分開，之後才知道容器裡跑的是哪一次訓練。

---

## 已知限制

| 限制 | 理由 |
|---|---|
| 只在本機訓練 | 不用雲端 GPU，訓練腳本、權重、報告在同一台機器——「換模型只需替換一個檔案」才站得住 |
| 沿用 YOLOv8 架構 | RF-DETR 等較新架構可能更好，但現有流程尚未走順，換架構會引入新的不確定性 |
| 沒有自動化測試 | 訓練腳本的正確性靠評估結果驗證，不靠單元測試 |
| 訓練環境與推論環境分離 | 本 repo 的 `.venv` 是 CUDA 版，`model-service` 維持 CPU 版——容器裡沒有 GPU |
| 未量化標註缺失的規模 | 僅逐框查證 2 張影像，未統計 213 張 test 集的比例——見評估報告第八節 |

---

## 授權

**AGPL-3.0** —— 因使用 Ultralytics YOLO。

- 整個 repo 必須保持公開
- 自訓練的權重**也受 AGPL 約束**
- 商業使用需向 Ultralytics 取得 Enterprise License

若之後要商業化，替代方案是換用授權較寬鬆的偵測模型——在 vision-detect 的架構下改動成本很低（推論服務是獨立容器）。

### 資料集

資料集不隨 repo 散布（`datasets/` 不進版控），各自授權見 [`docs/SOURCES.md`](docs/SOURCES.md)。

| 資料集 | 授權 | 署名 |
|---|---|---|
| Hard Hat Sample | Public Domain (CC0) | 不要求 |
| **PPE Detection** | **CC BY 4.0** | **要求** |

PPE Detection 由 **SDP** 發布於 [Roboflow Universe](https://universe.roboflow.com/sdp-lfigk/ppe-detection-ozhfb)，  
採 CC BY 4.0 授權。評估報告與任何使用該資料集的產出都需標註來源。
