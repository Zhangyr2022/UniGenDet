SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"

set -euo pipefail
set -x

# Runs fake-image detection evaluation. The testing entrypoint is placed under eval/det.
MODEL_PATH=${MODEL_PATH:-/path/to/project/pretrained}
CKPT_PATH=${CKPT_PATH:-/path/to/project/results_train_fakevlm_ema0.95_label_balanced_generation_aug/0001200/ema.safetensors}
TEST_JSON_PATH=${TEST_JSON_PATH:-/path/to/datasets/fakevlm/data_json/test.json}
IMAGE_ROOT=${IMAGE_ROOT:-/path/to/datasets/fakevlm/test}
OUTPUT_DIR=${OUTPUT_DIR:-/path/to/project/aug_fakevlm_test_results_multi_gpu}
FINAL_RESULTS_FILE=${FINAL_RESULTS_FILE:-}

python eval/det/fakevlm_test.py \
	--model_path "$MODEL_PATH" \
	--ckpt_path "$CKPT_PATH" \
	--test_json_path "$TEST_JSON_PATH" \
	--image_root "$IMAGE_ROOT" \
	--output_dir "$OUTPUT_DIR" \
	${FINAL_RESULTS_FILE:+--final_results_file "$FINAL_RESULTS_FILE"}