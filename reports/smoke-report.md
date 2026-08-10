# 可行性驗證報告

> 執行日期：2026-08-10
> 結論：**七項全部通過。** 過程中發現並修補 `vision-detect` 的一個配置缺口。

用 100 張的小資料集走完整輪。**目的不是得到好模型，是確認環境與流程沒問題。**

與第一階段「先用 COCO 打通鏈路」同一個邏輯：先讓最不確定的部分變確定。

---

## 一、結果

| 項目 | 結果 | 備註 |
|---|---|---|
| ① CUDA 環境正常 | ✅ | torch 2.11.0+cu128 |
| ② Roboflow 匯出格式能被讀取 | ✅ | |
| ③ `data.yaml` 路徑設定正確 | ✅ | 需改為絕對路徑，見四-① |
| ④ 訓練在 GPU 上跑起來，VRAM 沒爆 | ✅ | 20 epochs / 41 秒 |
| ⑤ 權重塞進 `model-service` 容器 | ⚠️→✅ | **需先補 compose 缺口**，見三 |
| ⑥ `/labels` 與 `model_version` 更新 | ✅ | 三類別，順序一致 |
| ⑦ 儀表板顯示新類別 | ✅ | 篩選 UI 未測，見七-③ |

---

## 二、環境與資料

### 環境

```
Python      3.12.3
torch       2.11.0+cu128
ultralytics 8.4.116
GPU         NVIDIA GeForce RTX 3080 · 12.0 GB · sm_86
```

> ⚠️ **顯示卡型號與既有文件不符。** 驅動回報 RTX 3080（12 GB 版本），
> `HANDOFF` 與 README 記載為 RTX 3080 Ti。兩者 CUDA 核心數不同（8704 vs 10240），
> 訓練時間會有落差。**文件的規格表應更正為實際型號。**

### 資料集

| 項目 | 內容 |
|---|---|
| 名稱 | Hard Hat Sample（v1-raw，未套用預處理與增強） |
| 來源 | https://universe.roboflow.com/kalidevi/hard-hat-sample/dataset/1 |
| 原始出處 | Northeastern University - China，哈佛 Dataverse DOI: 10.7910/DVN/7CBGOS |
| 授權 | Public Domain（CC0） |
| 規模 | 100 張 · train 70 / val 20 / test 10 |
| 類別 | head、helmet、person |

**類別分布嚴重失衡**（`dataset_stats.py` 輸出）：

| 類別 | train | val | test | 全體佔比 |
|---|---|---|---|---|
| head | 64 | 18 | 8 | 24.5% |
| helmet | 192 | 45 | 50 | 72.9% |
| person | 7 | 2 | **0** | 2.6% |

**`person` 在 test 集完全沒有樣本** —— 該類別的 mAP 會是 0 或 NaN，這是資料的必然，不是模型的問題。

**框偏小**：train 中位相對面積 0.0087（640 輸入下約 60 px 見方），test 更小僅 0.0041。
屬小物件偵測任務，正式訓練時輸入解析度可能要調高。

> **先跑統計的價值**：若未事先知道，訓練完看到 `person` mAP = 0 會誤以為程式壞了。

---

## 三、發現的缺口：compose 未支援換模型

**這是本次驗證最重要的產出。**

`README(zh).md` 寫著：

> 模型權重在建置階段內建於映像。換模型時可用 volume 掛載覆寫，並修改
> `MODEL_WEIGHTS` 環境變數 —— 不需重新建置映像。

**驗證前這句話不成立。** 三個必要條件一個都沒到位：

| 條件 | 驗證前狀態 |
|---|---|
| `docker-compose.yml` 有 volume 掛載 | ❌ 第 68–70 行整段被註解 |
| compose 傳入 `MODEL_WEIGHTS` | ❌ 只有 `MODEL_VERSION` 等三個 |
| `model-service/weights/` 目錄存在 | ❌ `yolov8n.pt` 直接放在根目錄 |

**Python 端是對的** —— `config.py` 的 `get_weights()` 早就在讀 `MODEL_WEIGHTS`，
`/labels` 端點也存在（`main.py:80`）。缺的只有配置。

### 修補內容（`vision-detect/docker-compose.yml`）

```yaml
    environment:
      MODEL_WEIGHTS: ${MODEL_WEIGHTS:-yolov8n.pt}    # ← 新增
      MODEL_VERSION: ${MODEL_VERSION:-yolov8n-coco}
      CONFIDENCE_THRESHOLD: ${CONFIDENCE_THRESHOLD:-0.25}
      MAX_CONCURRENCY: ${MAX_CONCURRENCY:-1}

    volumes:                                          # ← 解除註解
      - ./model-service/weights:/app/weights:ro
```

預設值保持 `yolov8n.pt`，不設 `MODEL_WEIGHTS` 時行為與修補前一致，不影響既有部署。

### 這一項的意義

**若先訓練 2,114 張再發現，代價是幾十小時。** 可行性驗證的七項裡，
前四項只驗 ML 環境，後三項驗的是**第一階段架構主張是否成立** —— 那才是這一步真正的目的。

詳細操作紀錄見 `docs/export_to_service-實錄.md`。

---

## 四、踩到的坑

### ① `data.yaml` 的相對路徑

Roboflow 匯出的 `train: ../train/images`，`../` 假設你從資料夾內某層執行。
從 repo 根目錄跑會解析到 `datasets/train/images`。

改為絕對路徑加 `path` 鍵，用正斜線（反斜線在 YAML 是跳脫字元）：

```yaml
path: C:/dev/PR_vision-detect-ml/vision-detect-ml/datasets/hardhat-sample-100
train: train/images
val: valid/images
test: test/images
```

### ② Ultralytics 的 `datasets_dir` 會攔截相對路徑

Ultralytics 解析 `data` 的相對路徑時，先看 `settings.json` 的 `datasets_dir`，不是工作目錄。

`train.py` 因此在傳出去之前把 `data` 轉成絕對路徑 —— config 可繼續寫相對路徑，實際生效的是絕對路徑。

### ③ `MODEL_WEIGHTS` 必須是容器內絕對路徑

只寫檔名的話，Ultralytics 當成官方預訓練權重的名稱去 GitHub 下載。

```properties
MODEL_WEIGHTS=/app/weights/hardhat-100-v1.pt    # ✅
MODEL_WEIGHTS=hardhat-100-v1.pt                 # ❌ 會去網路下載
```

症狀：`docker compose logs model` 出現 `Downloading .../yolov8n.pt`。

### ④ PS 5.1 的 `Set-Content` 預設不是 UTF-8

用管線改寫含中文註解的 YAML 時，`Set-Content` 寫成 CP950，Python 用 UTF-8 讀就炸：

```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xaf in position 3
```

**讀寫都要明確加 `-Encoding UTF8`。** 這比 `HANDOFF` 記載的「中文註解顯示為亂碼」更嚴重 ——
那個只是顯示問題，這個會讓程式讀不了檔案。

### ⑤ `Add-Content` 重複執行造成 `.env` 重複鍵

執行兩次後 `MODEL_VERSION` 出現三次。最後一個生效，所以**碰巧正確** ——
但這靠位置僥倖成立，重排順序就壞，而且不會報錯。

與 `.env.docker.example` 開頭那段警語同一類失敗：**靜靜套用錯的值。**
修法是用編輯器改，不要追加。

### ⑥ `docker compose down` 之後不能只起 model

`down` 移除全部容器，指定 `model` 就只起一個 —— nginx 沒起來，port 80 連不上。

---

## 五、訓練結果

`configs/smoke.yaml`：yolov8n、20 epochs、640、batch 16、device 0、workers 8、seed 0。

**41.3 秒跑完 20 epochs。** GPU 確實在工作 —— CPU 跑同樣的量會是好幾分鐘。

最後一個 epoch：

| 指標 | 值 |
|---|---|
| precision | 0.884 |
| recall | 0.348 |
| mAP@50 | 0.494 |
| mAP@50-95 | 0.321 |

**形狀很典型**：precision 高、recall 低 —— 模型很保守，找到的多半是對的，但漏掉三分之二。
70 張訓練集下這是必然，它只學會了最明顯的那些安全帽。

`results.csv` 只給整體 mAP，看不出 `person` 那 7 個框的表現。分類別數據要靠 `evaluate.py`。

---

## 六、端到端驗證

`/labels` 回傳（從容器內打，模型服務對外不可達）：

```json
{"model_version":"hardhat-100-v1","count":3,
 "labels":[{"class_id":0,"label":"head"},
           {"class_id":1,"label":"helmet"},
           {"class_id":2,"label":"person"}]}
```

**類別與順序與 `.pt` 完全一致。** 順序決定 `class_id`，對不上的話資料庫舊紀錄與新模型的語意就對不上。

送圖結果：

| 來源 | 張數 | 結果 |
|---|---|---|
| test 集 | 1 | `detections: []`，`modelVersion` 正確 |
| train 集 | 5 | 全部認出 `helmet`（其中一張 4 個） |

test 集空結果**不算失敗** —— recall 0.35 且該集框特別小。
第 ⑦ 項驗的是換模型的鏈路，判定看 `modelVersion` 是否為 `hardhat-100-v1`。

儀表板即時新增五列，標籤顯示為 `helmet`，舊紀錄仍保留 COCO 標籤與各自的 `modelVersion` ——
**換模型後結果來源可追溯**，這是第一階段設計的回報。

### 一個要註明的數字

新紀錄推論耗時 62–82 ms，舊紀錄（手機拍攝）220–260 ms。

**差異來自輸入尺寸**（500×666 vs 3120×4160），不是模型變快。
寫進評估報告時必須註明，否則會被誤讀成 fine-tune 提升了效能。

---

## 七、待處理

**① `vision-detect` 的文件需更新。**
`README(zh).md` 的「不需重新建置映像」現在才成立；
`.env.docker.example` 應加註 `MODEL_WEIGHTS` 的路徑格式要求，
但本身要保持 COCO 預設值 —— 別人 clone 下來時 `model-service/weights/` 是空的，
指向不存在的檔案會啟動失敗。

**② 兩份文件的硬體規格要更正。**
RTX 3080 Ti → RTX 3080（見二）。

**③ 前端類別篩選是否已接 `/labels` 未確認。**
`HANDOFF` 的下一步第 ④ 項記載「前端篩選功能未實作」。
本次僅確認標籤顯示正確，篩選 UI 的行為未測。

**④ `:ro` 唯讀掛載的潛在問題。**
Ultralytics 載入舊格式 checkpoint 時可能嘗試寫入同目錄。
本次使用 8.4.116 剛訓練的權重未觸發。日後若報寫入錯誤，拿掉 `:ro` 即可。

---

## 八、下一步

可行性驗證完成，環境與流程確認可用。進入第 ② 步：

換成 2,114 張的 PPE 資料集正式訓練（6 類：Gloves、Hard_hat、Mask、Person、Safety_boots、Vest）。
下載前依 `docs/階段規劃.md` 的三項檢查：授權標示、標註品質抽查、類別分布。

在那之前需要 `evaluate.py` —— 本次只拿到整體 mAP，
分類別的 PR 曲線與混淆矩陣才是評估報告的主體。
