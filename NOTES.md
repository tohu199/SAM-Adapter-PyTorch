`cod-sam-vit-l.yaml` を基準に SAM3 対応を確認し、1 epoch の学習まで通しました。

## SAM3 対応の確認結果

| 項目 | 状態 |
|---|---|
| **設定** (`cod-sam-vit-l.yaml`) | `inp_size=1008`, ViT-L (1024/24), `sam_checkpoint=./pretrained/sam3.pt` |
| **学習コード** (`train.py`) | `detector.backbone.*` → `image_encoder.*` の SAM3 形式ローダー |
| **モデル** (`models/sam.py`) | 1008 入力で forward 可能（312M params） |
| **チェックポイント** | `detector.backbone` キー 399 個をロード（`Load result: 0 missing keys`） |

検証コマンド:

```bash
source .venv/bin/activate
python scripts/verify_sam3_config.py --config configs/cod-sam-vit-l.yaml
python scripts/verify_sam3_codebase.py
```

## 実行結果

```bash
bash run_train.sh   # configs/cod-sam-vit-l.yaml
```

- 学習 2 step（スモーク用データ 2 枚 × `batch_size=1`）完了
- 出力: `save/_cod-sam-vit-l/model_epoch_last.pth`（約 1.2GB）

## 修正した不具合（SAM3 ブランチ側の欠落）

1. **`train.py`**: `model(inp, gt)` → `set_input` + `optimize_parameters`（SAM2 ブランチと同じ）
2. **`train.py`**: val の `name` / `shape` でログが落ちる問題を修正
3. **`train.py`**: val 時はテンソルのみ `.cuda()` に変更
4. **`test.py`**: 存在しない `models.sam2` import を `torchvision.transforms` に変更
5. **`models/mmseg`**: mmcv 不要で SAM モジュールを import 可能に

## 注意点

### 1. `sam3.pt`（今回はスモーク用）

本物の [facebook/sam3](https://huggingface.co/facebook/sam3) は gated モデルで、未ログインでは取得できません。今回はパイプライン確認用に:

```bash
python scripts/create_smoke_sam3_ckpt.py   # テスト専用・精度は出ない
```

本番学習前に:

```bash
huggingface-cli login
python scripts/download_sam3_ckpt.py
```

### 2. データセット

`cod-sam-vit-l.yaml` のパスを `./load/codDataset/...` に変更済み。本番は COD10K 等を README 通り `./load` に配置してください。

### 3. RTX 3070（8GB）と validation

1008 + ViT-L では **validation で OOM** になりました。8GB GPU では `epoch_val: null` にするか、検証は `python test.py` を別途・低解像度で実行するのが現実的です。

### 4. `epoch_max`

スモーク後に `epoch_max: 100` / `epoch_val: 1` に戻してあります。

---

**まとめ**: コードベースは `cod-sam-vit-l.yaml` の SAM3 設定と整合しており、学習パイプラインは動作確認済みです。本番では公式 `sam3.pt` の取得と実データセットの配置が必要です。HF ログイン後に公式重みで再実行する手順も進められます。