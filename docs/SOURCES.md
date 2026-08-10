# 資料集來源紀錄

> 下載當下就記錄。授權標示原文見各資料集資料夾內的 `README.dataset.txt`。

## Hard Hat Sample

| 項目 | 內容 |
|---|---|
| 用途 | 可行性驗證 |
| 本機路徑 | `datasets/hardhat-sample-100/` |
| 規模 | 100 張（train 70 / valid 20 / test 10） |
| 類別 | head、helmet、person（3 類） |
| 標註格式 | bounding box |
| 授權 | **Public Domain（CC0）** —— 不要求署名 |
| 下載頁 | https://universe.roboflow.com/kalidevi/hard-hat-sample/dataset/1 |
| 匯出版本 | v1-raw（未套用預處理與增強） |
| 原始出處 | Northeastern University - China<br>哈佛 Dataverse DOI: 10.7910/DVN/7CBGOS |
| 下載日期 | 2026-08-09 |

## PPE Detection

| 項目 | 內容 |
|---|---|
| 用途 | 正式練習 |
| 本機路徑 | `datasets/ppe-2114/` |
| 規模 | 2,114 張（train 1476 / valid 425 / test 213） |
| 類別 | Gloves、Hard_hat、Mask、Person、Safety_boots、Vest（6 類） |
| 標註格式 | **多邊形**（Ultralytics 訓練偵測模型時自動轉外接框） |
| 授權 | **CC BY 4.0 —— 要求署名** |
| 下載頁 | https://universe.roboflow.com/sdp-lfigk/ppe-detection-ozhfb/dataset/13 |
| 匯出版本 | v13i.yolov8 |
| 作者 | SDP |
| 下載日期 | 2026-08-10 |

### 署名（CC BY 4.0 要求）

論文或報告引用時：

```bibtex
@misc{ ppe-detection-ozhfb_dataset,
  title = { PPE DETECTION Dataset },
  type = { Open Source Dataset },
  author = { SDP },
  howpublished = { \url{ https://universe.roboflow.com/sdp-lfigk/ppe-detection-ozhfb } },
  url = { https://universe.roboflow.com/sdp-lfigk/ppe-detection-ozhfb },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2024 },
  month = { apr }
}
```

---

## 兩個資料集的差異（實測，非文件記載）

| | Hard Hat Sample | PPE Detection |
|---|---|---|
| 標註格式 | bbox（每行 5 值） | 多邊形（每行 7+ 值） |
| 類別均衡度 | 極差（person 2.6%） | 良好（10.8%–22.0%） |
| 框大小 P90 | 0.0508 | 0.2158 |
| 空標註 | 0 張 | 109 張（5.2%，負樣本） |

**匯出格式不同，而 Roboflow 頁面不會標示。** `dataset_stats.py` 原本只支援 bbox，
遇到多邊形直接 `ValueError`。下載新資料集後先跑統計，能提早發現這類差異。