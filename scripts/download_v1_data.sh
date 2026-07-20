#!/usr/bin/env bash
set -Eeuo pipefail

# 從 EvidenceGap repo 根目錄執行。
ROOT_DIR="$(pwd)"
DATA_DIR="${ROOT_DIR}/data/raw"

PROXY="http://127.0.0.1:7899"

CIVICFACT_DIR="${DATA_DIR}/civicfact"
CLINIFACT_DIR="${DATA_DIR}/clinifact"

CIVICFACT_BASE="https://raw.githubusercontent.com/creisle/civicfact/master/data_builder/builds/civicfact-2025.03.25"
CLINIFACT_BASE="https://raw.githubusercontent.com/ds4dh/CliniFact/main/data/processed/primary_outcome_publication_dataset"

export http_proxy="${PROXY}"
export https_proxy="${PROXY}"
export HTTP_PROXY="${PROXY}"
export HTTPS_PROXY="${PROXY}"

mkdir -p "${CIVICFACT_DIR}" "${CLINIFACT_DIR}"

download() {
    local url="$1"
    local output="$2"

    echo "Downloading: ${url}"
    curl \
        --fail \
        --location \
        --show-error \
        --retry 5 \
        --retry-delay 2 \
        --continue-at - \
        --proxy "${PROXY}" \
        --output "${output}" \
        "${url}"
}

echo "=== Download CIViC-Fact ==="

download \
    "${CIVICFACT_BASE}/data.jsonl.gz" \
    "${CIVICFACT_DIR}/data.jsonl.gz"

# 保留原始 CIViC dump，之後檢查來源資料時可能會用到。
download \
    "${CIVICFACT_BASE}/civic-dump.jsonl.gz" \
    "${CIVICFACT_DIR}/civic-dump.jsonl.gz"

echo "=== Download CliniFact ==="

for split in train validation test; do
    download \
        "${CLINIFACT_BASE}/${split}_set.csv" \
        "${CLINIFACT_DIR}/${split}_set.csv"
done

echo "=== Check compressed files ==="

gzip -t "${CIVICFACT_DIR}/data.jsonl.gz"
gzip -t "${CIVICFACT_DIR}/civic-dump.jsonl.gz"

echo "=== Check file contents ==="

python3 - "${CIVICFACT_DIR}" "${CLINIFACT_DIR}" <<'PY'
import csv
import gzip
import json
import sys
from pathlib import Path

civicfact_dir = Path(sys.argv[1])
clinifact_dir = Path(sys.argv[2])


def check_jsonl_gz(path: Path) -> int:
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path}: invalid JSON at line {line_number}: {exc}"
                ) from exc
            count += 1
    return count


def check_csv(path: Path) -> tuple[int, list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"{path}: CSV header is missing")

        count = sum(1 for _ in reader)
        return count, reader.fieldnames


civicfact_count = check_jsonl_gz(civicfact_dir / "data.jsonl.gz")
civic_dump_count = check_jsonl_gz(civicfact_dir / "civic-dump.jsonl.gz")

print(f"CIViC-Fact examples: {civicfact_count}")
print(f"CIViC dump records:   {civic_dump_count}")

for split in ("train", "validation", "test"):
    path = clinifact_dir / f"{split}_set.csv"
    count, columns = check_csv(path)

    print(f"CliniFact {split:10s}: {count} rows")
    print(f"  columns: {', '.join(columns)}")
PY

echo "=== Write checksums ==="

(
    cd "${CIVICFACT_DIR}"
    sha256sum data.jsonl.gz civic-dump.jsonl.gz > SHA256SUMS
)

(
    cd "${CLINIFACT_DIR}"
    sha256sum train_set.csv validation_set.csv test_set.csv > SHA256SUMS
)

cat > "${DATA_DIR}/DOWNLOAD_INFO.txt" <<EOF
Downloaded through proxy: ${PROXY}

CIViC-Fact:
  Repository: https://github.com/creisle/civicfact
  Build: civicfact-2025.03.25

CliniFact:
  Repository: https://github.com/ds4dh/CliniFact
  Dataset: primary_outcome_publication_dataset
EOF

echo
echo "Download completed."
echo "Data directory: ${DATA_DIR}"
du -sh "${CIVICFACT_DIR}" "${CLINIFACT_DIR}"
