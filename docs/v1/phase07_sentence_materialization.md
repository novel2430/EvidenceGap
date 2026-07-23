# EvidenceGap V1 Phase 07.1：Runtime Sentence Materialization

本阶段建立 Phase 04 Article Retrieval 与 Phase 05 Evidence Sentence Retrieval 之间的稳定桥接：

```text
Runtime Article
→ title / abstract source segments
→ Stanza GENIA sentence segmentation
→ stable RuntimeSentence artifacts
```

它不执行文章检索、sentence retrieval 或 stance 判断。

## 冻结方案

```text
library    = stanza 1.14.0
language   = en
processor  = tokenize only
package    = genia
device     = cuda:0 by default
runtime download = disabled
```

Title 在存在时固定作为一个独立 RuntimeSentence。Abstract 或结构化 sections 才交给 Stanza。当前 Phase 02 Article Corpus 的兼容字段 `text` 会被视为 abstract body；系统不会猜测 title 边界。

## 输入格式

支持：

```text
.json
.jsonl
.parquet
```

每条文章至少需要：

```json
{
  "article_id": "pmid:123",
  "title": "Optional title",
  "abstract": "Optional abstract"
}
```

也支持：

```json
{
  "article_id": "pmid:123",
  "sections": [
    {"section": "methods", "text": "..."},
    {"section": "results", "text": "..."}
  ]
}
```

以及当前语料兼容形式：

```json
{
  "article_id": "pmid:123",
  "text": "Existing Phase 02 article text"
}
```

额外 retrieval/reranking 字段会保存在 `source_metadata_json`，供后续 Phase 07 adapters 使用。

## 安装

项目原有 CUDA PyTorch 环境不变，再安装：

```bash
pip install -r requirements/v1-phase07.txt
```

## 下载模型

模型下载和正式运行分离。正式运行不会静默联网：

```bash
python scripts/run_v1_phase07.py download-sentence-model \
  --root .
```

下载器默认使用 `--download-source auto`：先尝试 Hugging Face；若 Hub metadata/Xet 端点不可达，会自动改用 Stanford 官方下载服务器。也可以直接指定：

```bash
python scripts/run_v1_phase07.py download-sentence-model \
  --root . \
  --download-source stanford
```

默认模型目录：

```text
models/v1/stanza/
```

## GPU 环境验收

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_v1_phase07.py check-sentence-runtime \
  --root . \
  --device cuda:0
```

输出应包含：

```text
status = PASS
splitter.actual_device = cuda:0
torch.cuda_available = true
model_load_seconds
sample_inference_seconds
sample_sentences
```

需要明确允许 CUDA 初始化失败时退回 CPU，才增加：

```bash
--allow-cpu-fallback
```

默认不静默 fallback。

## 手动 Materialization 验收

仓库提供两篇示例文章：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_v1_phase07.py materialize-sentences \
  --root . \
  --input-path data/examples/v1/phase07_runtime_articles.jsonl \
  --run-name stanza_genia_smoke \
  --device cuda:0 \
  --force
```

输出目录：

```text
artifacts/v1/pipeline/runtime_sentences/stanza_genia_smoke/
├── runtime_articles.parquet
├── runtime_sentences.parquet
├── runtime_sentences.jsonl
└── run_manifest.json
```

快速人工检查：

```bash
sed -n '1,20p' \
  artifacts/v1/pipeline/runtime_sentences/stanza_genia_smoke/runtime_sentences.jsonl
```

正式验证：

```bash
python scripts/run_v1_phase07.py validate-sentences \
  --root . \
  --artifact-dir \
    artifacts/v1/pipeline/runtime_sentences/stanza_genia_smoke
```

验证内容包括：

```text
manifest 与输出 checksum
sentence_id 唯一性
每篇文章 sentence_index 连续性
character offsets round-trip
splitter fingerprint 一致性
输入文章覆盖完整性
```

## 稳定性规则

`sentence_id` 会绑定：

```text
article_id
canonical source text checksum
Stanza/GENIA semantic identity
sentence character offsets
sentence text
```

但不会绑定：

```text
cuda:0 / cpu
batch size
本地 model directory
是否发生 CPU fallback
```

因此相同版本模型在不同执行设备上产生相同边界时，Sentence ID 不会漂移。设备和性能信息仍完整保存在 run manifest。

## 后续 Phase 07 复用方式

后续 Runtime Article Retrieval Adapter 只需输出受支持的 JSON/Parquet fields；Sentence Retrieval Adapter 直接读取：

```text
runtime_sentences.parquet
```

主要复用字段：

```text
article_id
pmid
article_rank
sentence_id
sentence_index
sentence_type
section
sentence_text
character_start
character_end
source_text_fingerprint
splitter_fingerprint
```

Phase 06 context window 可以按同一 `article_id` 的连续 `sentence_index` 取得前后句，不需要重新切句。

## Structured abstract header recovery

MedFact raw articles preserve PubMed-style section headings only as inline text,
not as `AbstractText Label` / `NlmCategory` metadata. In `section_mode=auto`,
Phase 07 therefore applies a frozen deterministic allowlist before Stanza:

```text
flat abstract with at least two known headers
→ section segments
→ header stored as section metadata
→ section body sent to Stanza independently
```

For example:

```text
INTERVENTION: ... vitamin D MEASUREMENTS: The primary outcome ...
```

becomes separate `intervention` and `measurements` segments. The literal header
is not included in `sentence_text`. Parsing activates only when at least two
allowlisted headers are present; otherwise the abstract remains one segment.
The parser identity is recorded as:

```text
phase07.structured-abstract-header.v1
```

This recovery is deterministic and approximate. It cannot restore text already
truncated in the MedFact source and does not replace future PubMed XML ingestion.
