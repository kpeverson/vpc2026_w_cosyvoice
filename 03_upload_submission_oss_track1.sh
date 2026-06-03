#!/bin/bash
# VoicePrivacy Track 1 submission uploader — Aliyun OSS (CN + global dual bucket)
#
# One AccessKey works for both buckets; pick target with OSS_TARGET or explicit bucket/region.
#
# Connectivity test (CN):
#   OSS_TARGET=cn OSS_ACCESS_KEY_ID=XXX OSS_ACCESS_KEY_SECRET=YYY OSS_TEAM=team-PSST1 \
#       ./03_upload_submission_oss.sh test
#
# Connectivity test (Singapore):
#   OSS_TARGET=global OSS_ACCESS_KEY_ID=XXX OSS_ACCESS_KEY_SECRET=YYY OSS_TEAM=team-PSST1 \
#       ./03_upload_submission_oss.sh test
#
# Test both buckets with the same key:
#   OSS_TARGET=cn OSS_ACCESS_KEY_ID=XXX OSS_ACCESS_KEY_SECRET=YYY OSS_TEAM=team-PSST1 \
#       ./03_upload_submission_oss.sh test-all
#
# Upload (example, Singapore):
#   OSS_TARGET=global OSS_ACCESS_KEY_ID=... OSS_ACCESS_KEY_SECRET=... OSS_TEAM=team-PSST1 \
#       ./03_upload_submission_oss.sh _myanon
#
# Beijing ECS (internal network, CN only):
#   OSS_TARGET=cn OSS_ENDPOINT=oss-cn-beijing-internal.aliyuncs.com ...
#
# Reinstall upload tools: rm .done-upload-tool

set -e

nj=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
[ -f ./env.sh ] && source ./env.sh

##################################################
# OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET — from team_accesskeys.csv
# OSS_TEAM — folder column, e.g. team-PSST1
# OSS_TARGET — cn (default) | global; ignored if OSS_BUCKET and OSS_REGION are both set
# OSS_BUCKET / OSS_REGION / OSS_ENDPOINT — override defaults (see team_accesskeys.csv)
# OSS_PARALLEL / OSS_PART_SIZE — multipart upload tuning (large archives)
##################################################
OSS_TARGET="${OSS_TARGET:-cn}"
OSS_BUCKET="${OSS_BUCKET:-}"
OSS_REGION="${OSS_REGION:-}"
OSS_ENDPOINT="${OSS_ENDPOINT:-}"
OSS_PARALLEL="${OSS_PARALLEL:-32}"
OSS_PART_SIZE="${OSS_PART_SIZE:-64M}"

resolve_oss_target() {
  local target=$1

  if [[ -n "$OSS_BUCKET" && -n "$OSS_REGION" ]]; then
    case "$OSS_REGION" in
      ap-southeast-1) OSS_TARGET=global ;;
      *)              OSS_TARGET=cn ;;
    esac
  else
    OSS_TARGET=$target
    case "$target" in
      cn)
        OSS_BUCKET="${OSS_BUCKET:-vpc2026-cn}"
        OSS_REGION="${OSS_REGION:-cn-beijing}"
        ;;
      global)
        OSS_BUCKET="${OSS_BUCKET:-vpc2026-global}"
        OSS_REGION="${OSS_REGION:-ap-southeast-1}"
        ;;
      *)
        echo "Error: OSS_TARGET must be 'cn' or 'global' (got: ${target})"
        exit 1
        ;;
    esac
  fi

  if [[ -z "$OSS_ENDPOINT" ]]; then
    case "$OSS_REGION" in
      ap-southeast-1) OSS_ENDPOINT=oss-ap-southeast-1.aliyuncs.com ;;
      cn-beijing)     OSS_ENDPOINT="" ;;
    esac
  fi
}

print_oss_config() {
  echo "    target=${OSS_TARGET} bucket=${OSS_BUCKET} region=${OSS_REGION} endpoint=${OSS_ENDPOINT:-<default>}"
}

run_connectivity_test() {
  resolve_oss_target "$OSS_TARGET"
  print_oss_config

  export OSS_ACCESS_KEY_ID OSS_ACCESS_KEY_SECRET
  local oss_args=(--region "$OSS_REGION")
  [[ -n "$OSS_ENDPOINT" ]] && oss_args+=(-e "$OSS_ENDPOINT")

  local dest_dir="oss://${OSS_BUCKET}/${OSS_TEAM}"
  echo " -- [target=${OSS_TARGET}] Connectivity test: ${dest_dir}/ --"

  local tmpf
  tmpf=$(mktemp)
  echo "connectivity test $(date) target=${OSS_TARGET}" > "$tmpf"
  local testkey="${dest_dir}/.connectivity_test_${OSS_TARGET}_$(date +'%Y%m%d_%H%M%S').txt"
  "$OSSUTIL" "${oss_args[@]}" cp -f "$tmpf" "$testkey"
  "$OSSUTIL" "${oss_args[@]}" rm -f "$testkey"
  rm -f "$tmpf"
  echo " -- [target=${OSS_TARGET}] OK: ${dest_dir}/ is writable --"
}

# Select the anonymization data suffix (or test mode)
if [ -n "$1" ]; then
  anon_suffix=$1
else
  echo "Provide anon_suffix, or 'test' / 'test-all' for connectivity checks."
  exit 1
fi

if [[ -z "$OSS_ACCESS_KEY_ID" || -z "$OSS_ACCESS_KEY_SECRET" || -z "$OSS_TEAM" ]]; then
  echo "Error: OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET / OSS_TEAM must be set"
  exit 1
fi
OSS_TEAM="${OSS_TEAM// /-}"

# Install ossutil (one time)
mark=.done-upload-tool
if [ ! -f "$mark" ]; then
  echo " == Installing tools to upload dataset =="
  mkdir -p ./utils
  if ! command -v ossutil >/dev/null 2>&1; then
    V=2.2.1
    case "$(uname -s)" in Linux) OS=linux;; Darwin) OS=mac;; *) echo "Unsupported OS"; exit 1;; esac
    case "$(uname -m)" in x86_64|amd64) ARCH=amd64;; aarch64|arm64) ARCH=arm64;; *) echo "Unsupported arch"; exit 1;; esac
    PKG="ossutil-${V}-${OS}-${ARCH}"
    curl -fsSL "https://gosspublic.alicdn.com/ossutil/v2/${V}/${PKG}.zip" -o ./utils/ossutil.zip
    unzip -o -q ./utils/ossutil.zip -d ./utils
    cp "./utils/${PKG}/ossutil" ./utils/ossutil-bin
    chmod +x ./utils/ossutil-bin
  fi
  if ! command -v pigz >/dev/null 2>&1 && command -v micromamba >/dev/null 2>&1; then
    micromamba install -y -c conda-forge pigz pv tar || true
  fi
  touch "$mark"
fi

if command -v ossutil >/dev/null 2>&1; then OSSUTIL=ossutil; else OSSUTIL=./utils/ossutil-bin; fi

# ===== Connectivity test =====
if test "$anon_suffix" = "test"; then
  echo " -- Running connectivity test for team '${OSS_TEAM}' --"
  run_connectivity_test
  exit 0
fi

if test "$anon_suffix" = "test-all"; then
  echo " -- Testing CN + global buckets for team '${OSS_TEAM}' (same AccessKey) --"
  saved_bucket=$OSS_BUCKET saved_region=$OSS_REGION saved_endpoint=$OSS_ENDPOINT
  for OSS_TARGET in cn global; do
    OSS_BUCKET=$saved_bucket
    OSS_REGION=$saved_region
    OSS_ENDPOINT=$saved_endpoint
    echo ""
    run_connectivity_test
  done
  echo ""
  echo " -- All connectivity tests passed --"
  exit 0
fi

resolve_oss_target "$OSS_TARGET"
export OSS_ACCESS_KEY_ID OSS_ACCESS_KEY_SECRET
OSS_ARGS=(--region "$OSS_REGION")
[[ -n "$OSS_ENDPOINT" ]] && OSS_ARGS+=(-e "$OSS_ENDPOINT")
DEST_DIR="oss://${OSS_BUCKET}/${OSS_TEAM}"

# If a yaml config was passed, read the real anon_suffix from it
if [[ "$anon_suffix" == *yaml ]]; then
  echo " -- Config detected, reading 'anon_suffix' --"
  anon_suffix=$(python3 -c "from hyperpyyaml import load_hyperpyyaml; f = open('${anon_suffix}'); print(load_hyperpyyaml(f, None).get('anon_suffix', ''))")
fi
echo " -- [target=${OSS_TARGET}] Track 1 submission, anon suffix: '${anon_suffix}' --"
print_oss_config

# ===== Collect submission files =====
stuff_to_zip=""
results_exp=exp/results_summary/track1

file=${results_exp}/result_for_rank${anon_suffix}
[ ! -f "$file" ] && echo "File $file does not exist." && exit 1
file=${results_exp}/result_for_submission${anon_suffix}.zip
[ ! -f "$file" ] && echo "File $file does not exist." && exit 1
stuff_to_zip="${stuff_to_zip} ${results_exp}/result_for_rank${anon_suffix} ${results_exp}/result_for_submission${anon_suffix}.zip"

# Track 1 anonymized wav dirs (libri + IEMOCAP + train-clean-360, trials_mixed layout).
tuples=(
  data/libri_dev_enrolls${anon_suffix}              7324598
  data/libri_dev_trials_mixed${anon_suffix}         455295386
  data/libri_test_enrolls${anon_suffix}            86805881
  data/libri_test_trials_mixed${anon_suffix}      416608014
  data/IEMOCAP_dev${anon_suffix}                  418919757
  data/IEMOCAP_test${anon_suffix}                 388856264
  data/train-clean-360${anon_suffix}            41937610246
)
length=${#tuples[@]}
for ((i=0; i<length; i+=2)); do
  dir=${tuples[i]}
  [ ! -d "$dir" ] && echo "Directory $dir does not exist." && exit 1
  threshold=${tuples[i+1]}
  dir_size=$(du -sb "$dir" | cut -f1)
  if [ "$dir_size" -lt "$threshold" ]; then
    echo "Directory '$dir' size ($dir_size bytes) is not greater than $threshold bytes. The wavs must be in this folder for submission." && exit 1
  fi
  stuff_to_zip="${stuff_to_zip} ${dir}"
done

# ===== Pack =====
echo " -- [target=${OSS_TARGET}] Creating submission archive (using: $nj threads) --"
archive="submission_track1_${OSS_TARGET}${anon_suffix}.tar.gz"
if command -v pigz >/dev/null 2>&1; then
  tar --use-compress-program="pigz --best --processes $nj" -cf "$archive" $stuff_to_zip
else
  tar -czf "$archive" $stuff_to_zip
fi

# ===== Upload to OSS =====
remote_name="submission_track1_${OSS_TARGET}_${OSS_TEAM}${anon_suffix}_$(date +'%Y-%m-%d_%H-%M-%S').tar.gz"
echo " -- [target=${OSS_TARGET}] Uploading ($(du -sbh "$archive" | cut -f1)) to ${DEST_DIR}/${remote_name}"
echo "    target=${OSS_TARGET} parallel=${OSS_PARALLEL} part-size=${OSS_PART_SIZE} --"
"$OSSUTIL" "${OSS_ARGS[@]}" cp -f \
  --parallel "$OSS_PARALLEL" --part-size "$OSS_PART_SIZE" \
  "$archive" "${DEST_DIR}/${remote_name}"
echo " -- [target=${OSS_TARGET}] Upload finished: ${DEST_DIR}/${remote_name} --"
