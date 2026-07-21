"""
Standalone CosyVoice2 anonymization — runs in the 'cosyvoice' conda env.

Replicates CosyVoice2Pipeline without importing the VPC framework, so it
works with torch 2.3.x (cosyvoice env) rather than the venv's torch 2.8.x.

Usage:
    conda run -n cosyvoice -- python run_cosyvoice_anon_standalone.py \
        --config configs/track1/anon_cosyvoice2.yaml [--gpu 0] [--force]
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

PROSODY_RATE = 12.5  # Hz — prosody embeddings are downsampled to this rate

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config loading (handles VPC's HyperPyYAML !ref syntax)
# ---------------------------------------------------------------------------

def _ref_constructor(loader, node):
    return loader.construct_scalar(node)

yaml.add_constructor('!ref', _ref_constructor, Loader=yaml.SafeLoader)


def _resolve_refs(cfg):
    """Resolve !ref <key> substitutions in a VPC config dict."""
    def resolve(s, ctx):
        for _ in range(5):
            s2 = re.sub(r'<([^>]+)>', lambda m: str(ctx.get(m.group(1), '<' + m.group(1) + '>')), s)
            if s2 == s:
                break
            s = s2
        return s

    # Build flat context from top-level scalars
    ctx = {k: str(v) for k, v in cfg.items() if isinstance(v, (str, int, float, bool))}

    # Resolve top-level strings
    for k, v in list(cfg.items()):
        if isinstance(v, str):
            cfg[k] = resolve(v, ctx)
            ctx[k] = cfg[k]

    # Resolve modules dict using the resolved top-level ctx only.
    # Do NOT merge unresolved module values into the context — they would
    # override already-resolved top-level keys (e.g. anon_suffix).
    if isinstance(cfg.get('modules'), dict):
        for k, v in list(cfg['modules'].items()):
            if isinstance(v, str):
                cfg['modules'][k] = resolve(v, ctx)
    return cfg


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return _resolve_refs(cfg)


# ---------------------------------------------------------------------------
# Kaldi I/O helpers
# ---------------------------------------------------------------------------

def read_kaldi(path, values_as_string=False):
    """Read a Kaldi-format file into a dict. Returns {} if file missing."""
    result = {}
    p = Path(path)
    if not p.exists():
        return result
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                result[parts[0]] = parts[1] if values_as_string else parts[1].split()
            elif len(parts) == 1:
                result[parts[0]] = '' if values_as_string else []
    return result


def expand_datasets(cfg):
    """Return [(dataset_name, src_path)] from config datasets list."""
    data_dir = Path(cfg['data_dir'])
    entries = []
    for ds in cfg['datasets']:
        base = ds['data']
        enrolls = ds.get('enrolls', [])
        trials  = ds.get('trials',  [])
        if enrolls or trials:
            for suffix in enrolls + trials:
                name = base + suffix
                entries.append((name, data_dir / name))
        else:
            entries.append((ds['name'], data_dir / base))
    return entries


# ---------------------------------------------------------------------------
# Audio loading (supports hdf5: URIs)
# ---------------------------------------------------------------------------

def load_audio(wav_entry):
    """Load audio from a wav.scp entry (file path or hdf5: URI).
    Returns (tensor [1, T], sample_rate).
    """
    entry = wav_entry if isinstance(wav_entry, str) else ' '.join(wav_entry)
    if entry.startswith('hdf5:'):
        rest = entry[len('hdf5:'):]
        last_colon = rest.rfind(':')
        h5_file, key = rest[:last_colon], rest[last_colon + 1:]
        with h5py.File(h5_file, 'r') as f:
            data = f[key][:]
            sr = int(f[key].attrs.get('sample_rate', 16000))
        signal = torch.from_numpy(data).float()
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        return signal, sr
    return torchaudio.load(entry)


def write_temp_wav(wav_entry):
    """Write audio to a temp wav file; return (path, is_temp)."""
    entry = wav_entry if isinstance(wav_entry, str) else ' '.join(wav_entry)
    if not entry.startswith('hdf5:'):
        return entry, False
    import tempfile
    signal, sr = load_audio(entry)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp = f.name
    torchaudio.save(tmp, signal, sr)
    return tmp, True


# ---------------------------------------------------------------------------
# Speaker embedding helpers
# ---------------------------------------------------------------------------

def load_pool(spk_pool_path):
    """Return (pool_matrix_raw [S,192], pool_matrix_norm [S,192], pool_ids [S])."""
    raw = torch.load(spk_pool_path, map_location='cpu')
    pool_ids = list(raw.keys())
    mat = torch.stack([torch.tensor(v, dtype=torch.float32) for v in raw.values()])
    return mat, F.normalize(mat, dim=1), pool_ids


def load_gender_file(path) -> dict:
    """Load a Kaldi spk2gender file into {spk_id: 'm'|'f'} dict."""
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                out[parts[0]] = parts[1].lower()
    return out


def build_anon_embedding(src_emb, pool_raw, pool_norm, n_anon, anon_percentile=None):
    """Return [1, 192] raw-scale mean of n_anon pool speakers.

    If anon_percentile is None (default): average the n_anon speakers with the
    lowest cosine similarity to src_emb (original strategy).

    If anon_percentile is set (0-100): find the speaker at that percentile of
    distance from src_emb (e.g. 90 → farther than 90% of pool), then average
    the n_anon speakers closest to that anchor. This keeps the target cluster
    coherent while still being far from the source.
    """
    src_n = F.normalize(src_emb.unsqueeze(0), dim=1)
    sims_to_src = (src_n @ pool_norm.T).squeeze(0)  # [S], higher = closer

    if anon_percentile is None:
        _, idx = torch.topk(sims_to_src, n_anon, largest=False)
    else:
        # Rank speakers from farthest to closest (ascending similarity)
        ranked = torch.argsort(sims_to_src)  # index 0 = farthest
        anchor_rank = int(anon_percentile / 100.0 * (len(ranked) - 1))
        anchor_idx = ranked[anchor_rank]
        # Find n_anon speakers closest to the anchor
        anchor_n = pool_norm[anchor_idx].unsqueeze(0)
        sims_to_anchor = (anchor_n @ pool_norm.T).squeeze(0)
        _, idx = torch.topk(sims_to_anchor, n_anon, largest=True)

    return pool_raw[idx].mean(dim=0, keepdim=True)  # [1, 192]


def select_anon_embedding(src_emb, src_spk, src_gender, pool_raw, pool_norm, pool_ids,
                          pool_genders, n_anon, anon_percentile, anon_method,
                          gender_mode, rng):
    """Unified embedding selector: applies gender filtering then chooses method.

    gender_mode: 'all' | 'same' | 'opposite'  (relative to src_gender)
    anon_method: 'farthest' (default) | 'random'
    rng: numpy.random.Generator (used only for 'random' method)
    The source speaker is never selected, even if they appear in the pool.
    """
    # Build candidate index list, excluding the source speaker
    candidate_ids = [
        i for i, pid in enumerate(pool_ids) if pid != src_spk
    ]

    # --- gender filtering ---
    if gender_mode != 'all' and pool_genders and src_gender:
        if gender_mode == 'same':
            candidate_ids = [i for i in candidate_ids if pool_genders.get(pool_ids[i]) == src_gender]
        else:  # opposite
            candidate_ids = [i for i in candidate_ids
                             if pool_genders.get(pool_ids[i]) is not None
                             and pool_genders.get(pool_ids[i]) != src_gender]
        if not candidate_ids:
            print(f'  Warning: no {gender_mode}-gender speakers in pool for '
                  f'src_gender={src_gender!r} (excluding self), falling back to full pool')
            candidate_ids = [i for i, pid in enumerate(pool_ids) if pid != src_spk]

    if not candidate_ids:
        candidate_ids = list(range(len(pool_ids)))  # last-resort fallback

    t = torch.tensor(candidate_ids)
    filtered_raw  = pool_raw[t]
    filtered_norm = pool_norm[t]

    # --- method ---
    if anon_method == 'random':
        idx = int(rng.integers(0, len(filtered_raw)))
        return filtered_raw[idx].unsqueeze(0)  # [1, 192]
    else:
        return build_anon_embedding(src_emb, filtered_raw, filtered_norm, n_anon, anon_percentile)


def presplit_asr_text(text, max_words=40):
    """Split punctuation-free ASR text into chunks before passing to CosyVoice2.

    split_paragraph() inside text_normalize only splits on punctuation marks.
    ASR transcripts have none, so the entire utterance becomes one segment and
    the LLM must generate hundreds of speech tokens at once — well outside its
    training distribution — causing truncation or garbled output. Pre-splitting
    on word boundaries keeps each chunk in the model's reliable operating range.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def load_asr_transcripts(asr_text_dir: str) -> dict:
    """Load all Kaldi text files under asr_text_dir into a merged utt->transcript dict."""
    merged = {}
    root = Path(asr_text_dir)
    for text_file in root.rglob('text'):
        entries = read_kaldi(text_file, values_as_string=True)
        merged.update(entries)
    print(f'  Loaded {len(merged)} ASR transcripts from {asr_text_dir}')
    return merged


def load_alignments(dataset_path):
    """Load word-level alignments produced by align.py for a dataset.

    Returns {utt_id: [word_entry, ...]} where each entry is a dict with
    'unit', 'characters', 'start', 'end'. Delimiter entries ('word_N_delim')
    are included; callers should filter them when building time lookups.
    Returns an empty dict if no alignment file exists for this dataset.
    """
    align_file = Path(dataset_path) / 'align_input_rank0_alignments.json'
    if not align_file.exists():
        return {}
    with open(align_file) as f:
        raw = json.load(f)
    return {utt: v.get('words', []) for utt, v in raw.items()}


def split_text_and_prosody(text, utt, prosody_emb, utt_word_entries, max_words=40):
    """Split text and prosody at aligned word boundaries.

    For utterances within max_words, returns the full text and prosody unchanged.
    For longer utterances, splits both text and prosody at the same word boundary
    so each chunk receives the prosody frames that correspond to its words.
    Falls back to None prosody for any chunk whose alignment boundary is missing.

    Args:
        text: lowercase text string (space-separated words)
        utt: utterance ID (used only for debug context)
        prosody_emb: tensor [1, T, D] at PROSODY_RATE Hz, or None
        utt_word_entries: list of word alignment dicts for this utterance, or []
        max_words: maximum words per chunk

    Returns:
        (text_chunks, prosody_chunks) — parallel lists of equal length
    """
    words = text.split()
    if len(words) <= max_words:
        return [text], [prosody_emb]

    chunk_starts = list(range(0, len(words), max_words))
    text_chunks = [' '.join(words[s:s + max_words]) for s in chunk_starts]

    if prosody_emb is None or not utt_word_entries:
        return text_chunks, [None] * len(text_chunks)

    # Build 1-indexed word number → end time, skipping delimiter entries
    word_end_times = {}
    for entry in utt_word_entries:
        if '_delim' not in entry['unit']:
            word_num = int(entry['unit'].split('_')[1])
            word_end_times[word_num] = entry['end']

    prosody_chunks = []
    prev_frame = 0
    for i, start in enumerate(chunk_starts):
        if i == len(chunk_starts) - 1:
            prosody_chunks.append(prosody_emb[:, prev_frame:, :])
        else:
            # words[start : start+max_words] are word numbers start+1 .. start+max_words
            # in 1-indexed alignment; the last of these is start+max_words
            split_word_num = start + max_words
            if split_word_num in word_end_times:
                split_frame = min(
                    int(round(word_end_times[split_word_num] * PROSODY_RATE)),
                    prosody_emb.shape[1],
                )
                prosody_chunks.append(prosody_emb[:, prev_frame:split_frame, :])
                prev_frame = split_frame
            else:
                # No alignment for this boundary — give up on prosody for all remaining chunks
                prosody_chunks.append(None)
                prosody_chunks.extend([None] * (len(chunk_starts) - len(prosody_chunks)))
                break

    return text_chunks, prosody_chunks


def load_utterance_embeddings(dataset_path, wav_scp, utt2spk, tts):
    """Load or extract per-utterance source embeddings. Returns {utt: tensor [192]}.

    Fallback chain:
      1. utt2embedding.pt  — pre-extracted per-utterance embeddings (preferred)
      2. spk2embedding.pt  — per-speaker averages; warns that all utterances from the
                             same speaker share the same source embedding
      3. on-the-fly extraction per utterance (slow)
    """
    utt_pt = dataset_path / 'utt2embedding.pt'
    if utt_pt.exists():
        print(f'  Loading per-utterance embeddings from {utt_pt}')
        raw = torch.load(utt_pt, map_location='cpu')
        return {utt: torch.tensor(v, dtype=torch.float32) for utt, v in raw.items()}

    spk_pt = dataset_path / 'spk2embedding.pt'
    if spk_pt.exists():
        print(f'  No utt2embedding.pt — falling back to per-speaker embeddings from {spk_pt}')
        print(f'  Warning: all utterances from the same speaker will share the same source embedding')
        raw = torch.load(spk_pt, map_location='cpu')
        spk2emb = {spk: torch.tensor(v, dtype=torch.float32) for spk, v in raw.items()}
        return {utt: spk2emb[spk] for utt, spk in utt2spk.items() if spk in spk2emb}

    print(f'  No utt2embedding.pt or spk2embedding.pt — extracting per-utterance on-the-fly (slow)')
    utt2emb = {}
    for utt, wav_path in tqdm(wav_scp.items(), desc='  Extracting embeddings'):
        try:
            path, is_tmp = write_temp_wav(wav_path)
            try:
                emb = tts.frontend._extract_spk_embedding(path).squeeze(0).cpu()
            finally:
                if is_tmp:
                    os.unlink(path)
            utt2emb[utt] = emb
        except Exception as e:
            print(f'  [warn] embedding failed for {utt}: {e}')
    return utt2emb


# ---------------------------------------------------------------------------
# Multi-GPU worker (module-level for pickle compatibility with spawn)
# ---------------------------------------------------------------------------

def _synthesis_worker(args):
    (utt_subset, cosyvoice_root, model_dir, prosody_encoder_path,
     gpu_index, utt2anon_emb_cpu, utt2text, utt2spk, wav_scp,
     utt2alignment, temp_h5_path, output_sr) = args

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)

    for p in [cosyvoice_root, os.path.join(cosyvoice_root, 'third_party', 'Matcha-TTS')]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from cosyvoice.cli.cosyvoice import CosyVoice2
    tts = CosyVoice2(model_dir, load_jit=False, load_trt=False,
                     prosody_encoder_path=prosody_encoder_path or '')
    device = tts.model.device
    use_prosody = bool(prosody_encoder_path)

    utt2anon_emb = {utt: emb.to(device) for utt, emb in utt2anon_emb_cpu.items()}

    with h5py.File(temp_h5_path, 'w') as hf:
        for utt in tqdm(utt_subset, desc=f'GPU {gpu_index}', position=gpu_index):
            text = utt2text.get(utt, '').lower()
            if not text or utt not in utt2anon_emb:
                continue
            try:
                prosody_emb = None
                if use_prosody and utt in wav_scp:
                    path, is_tmp = write_temp_wav(wav_scp[utt])
                    try:
                        prosody_emb = tts._wav_to_prosody_emb(path)
                    finally:
                        if is_tmp:
                            os.unlink(path)

                text_chunks, prosody_chunks = split_text_and_prosody(
                    text, utt, prosody_emb, utt2alignment.get(utt, [])
                )
                chunks = []
                for seg_text, seg_prosody in zip(text_chunks, prosody_chunks):
                    for seg in tts.frontend.text_normalize(seg_text, split=True):
                        tok, tok_len = tts.frontend._extract_text_token(seg)
                        model_input = {
                            'text': tok, 'text_len': tok_len,
                            'llm_embedding': utt2anon_emb[utt],
                            'flow_embedding': utt2anon_emb[utt],
                        }
                        if seg_prosody is not None:
                            model_input['prosody_emb'] = seg_prosody
                        for out in tts.model.tts(**model_input, stream=False, speed=1.0):
                            chunks.append(out['tts_speech'].cpu())

                if not chunks:
                    continue

                audio = torch.cat(chunks, dim=1)
                if tts.sample_rate != output_sr:
                    audio = torchaudio.functional.resample(audio, tts.sample_rate, output_sr)

                arr = audio.squeeze(0).numpy().astype(np.float32)
                ds = hf.create_dataset(utt, data=arr)
                ds.attrs['sample_rate'] = output_sr

            except Exception as e:
                print(f'[GPU {gpu_index}] failed for {utt}: {e}')

    return temp_h5_path


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synthesise_dataset(dataset_name, src_path, output_path, tts, pool_raw, pool_norm,
                       pool_ids, n_anon, cosyvoice_root, model_dir, prosody_encoder_path,
                       output_sr, gpu_ids, force, asr_transcripts=None,
                       anon_percentile=None, anon_method='farthest',
                       gender_mode='all', pool_genders=None, seed=0):
    """Anonymise one dataset directory."""
    print(f'\nProcessing: {dataset_name}')
    print(f'  {src_path} -> {output_path}')

    output_path.mkdir(parents=True, exist_ok=True)

    # Copy Kaldi metadata files (not wav.scp / audio.h5)
    skip = {'wav.scp', 'audio.h5'}
    for f in src_path.iterdir():
        if f.is_file() and f.name not in skip:
            dst = output_path / f.name
            if not dst.exists():
                shutil.copy2(f, dst)

    h5_path = output_path / 'audio.h5'

    # Check if already fully synthesised
    if not force and h5_path.exists():
        with h5py.File(h5_path, 'r') as hf:
            if hf.attrs.get('cosyvoice2_synthesized', False):
                print(f'  Already synthesised — skipping.')
                return

    wav_scp  = read_kaldi(src_path / 'wav.scp', values_as_string=True)
    utt2spk  = read_kaldi(src_path / 'utt2spk', values_as_string=True)
    utt2text = read_kaldi(src_path / 'text', values_as_string=True)
    if asr_transcripts:
        n_replaced = sum(1 for u in utt2text if u in asr_transcripts)
        print(f'  Using ASR transcripts for {n_replaced}/{len(utt2text)} utterances '
              f'(ground truth for remaining {len(utt2text) - n_replaced})')
        utt2text = {u: asr_transcripts.get(u, gt) for u, gt in utt2text.items()}

    utt2src_emb = load_utterance_embeddings(src_path, wav_scp, utt2spk, tts)

    src_spk2gender = load_gender_file(src_path / 'spk2gender') if gender_mode != 'all' else {}
    if gender_mode != 'all' and not src_spk2gender:
        print(f'  Warning: --gender={gender_mode} but no spk2gender in {src_path}; using all')
    if gender_mode != 'all' and not pool_genders:
        print(f'  Warning: --gender={gender_mode} but spk_pool_gender_path not set; using all')

    rng = np.random.default_rng(seed)

    # Build a per-utterance anon embedding — every utterance gets an independent draw.
    # The source embedding is per-utterance, so farthest/percentile methods use each
    # utterance's own embedding as the reference point.
    # Keep on CPU — workers move them to their own device.
    utt2anon_emb_cpu = {
        utt: select_anon_embedding(
            utt2src_emb[utt], utt2spk[utt], src_spk2gender.get(utt2spk[utt]),
            pool_raw, pool_norm, pool_ids, pool_genders, n_anon, anon_percentile,
            anon_method, gender_mode, rng,
        )
        for utt in utt2spk
        if utt in utt2src_emb
    }

    # Load already-done keys for resumability
    already_done = set()
    if not force and h5_path.exists():
        with h5py.File(h5_path, 'r') as hf:
            already_done = set(hf.keys())

    pending = [
        utt for utt in wav_scp
        if utt not in already_done
        and utt in utt2anon_emb_cpu
        and utt2text.get(utt, '').strip()
    ]

    utt2alignment = load_alignments(src_path)
    if utt2alignment:
        print(f'  Loaded word alignments for {len(utt2alignment)} utterances')
    else:
        print(f'  No alignment file found — long utterances will not use prosody')

    if not pending:
        print(f'  All {len(already_done)} utterances already done.')
    elif len(gpu_ids) == 1:
        _run_single_gpu(pending, h5_path, tts, utt2anon_emb_cpu, utt2text, utt2spk,
                        wav_scp, prosody_encoder_path, utt2alignment, output_sr, dataset_name)
    else:
        _run_multi_gpu(pending, output_path, h5_path, utt2anon_emb_cpu, utt2text, utt2spk,
                       wav_scp, cosyvoice_root, model_dir, prosody_encoder_path,
                       utt2alignment, output_sr, gpu_ids)

    # Write wav.scp
    with h5py.File(h5_path, 'r') as hf:
        lines = [f'{utt} hdf5:{h5_path.resolve()}:{utt}\n' for utt in sorted(hf.keys())]
    with open(output_path / 'wav.scp', 'w') as f:
        f.writelines(lines)
    print(f'  Wrote {len(lines)} entries to wav.scp')


def _run_single_gpu(pending, h5_path, tts, utt2anon_emb_cpu, utt2text, utt2spk,
                    wav_scp, prosody_encoder_path, utt2alignment, output_sr, desc):
    print(f'  Synthesising {len(pending)} utterances on 1 GPU...')
    device = tts.model.device
    use_prosody = bool(prosody_encoder_path)
    utt2anon_emb = {utt: emb.to(device) for utt, emb in utt2anon_emb_cpu.items()}

    with h5py.File(h5_path, 'a') as hf:
        hf.attrs['cosyvoice2_synthesized'] = True
        for utt in tqdm(pending, desc=f'  {desc}'):
            text = utt2text[utt].lower()
            emb  = utt2anon_emb[utt]
            try:
                prosody_emb = None
                if use_prosody and utt in wav_scp:
                    path, is_tmp = write_temp_wav(wav_scp[utt])
                    try:
                        prosody_emb = tts._wav_to_prosody_emb(path)
                    finally:
                        if is_tmp:
                            os.unlink(path)

                text_chunks, prosody_chunks = split_text_and_prosody(
                    text, utt, prosody_emb, utt2alignment.get(utt, [])
                )
                chunks = []
                for seg_text, seg_prosody in zip(text_chunks, prosody_chunks):
                    for seg in tts.frontend.text_normalize(seg_text, split=True):
                        tok, tok_len = tts.frontend._extract_text_token(seg)
                        model_input = {
                            'text': tok, 'text_len': tok_len,
                            'llm_embedding': emb, 'flow_embedding': emb,
                        }
                        if seg_prosody is not None:
                            model_input['prosody_emb'] = seg_prosody
                        for out in tts.model.tts(**model_input, stream=False, speed=1.0):
                            chunks.append(out['tts_speech'].cpu())

                if not chunks:
                    print(f'  [warn] no output for {utt}')
                    continue

                audio = torch.cat(chunks, dim=1)
                if tts.sample_rate != output_sr:
                    audio = torchaudio.functional.resample(audio, tts.sample_rate, output_sr)

                arr = audio.squeeze(0).numpy().astype(np.float32)
                if utt in hf:
                    del hf[utt]
                ds = hf.create_dataset(utt, data=arr)
                ds.attrs['sample_rate'] = output_sr

            except Exception as e:
                print(f'  [warn] synthesis failed for {utt}: {e}')


def _run_multi_gpu(pending, output_path, h5_path, utt2anon_emb_cpu, utt2text, utt2spk,
                   wav_scp, cosyvoice_root, model_dir, prosody_encoder_path,
                   utt2alignment, output_sr, gpu_ids):
    from torch.multiprocessing import Pool, set_start_method
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    n = len(gpu_ids)
    print(f'  Synthesising {len(pending)} utterances across {n} GPUs {gpu_ids}...')
    subsets   = [pending[i::n] for i in range(n)]
    temp_h5s  = [str(output_path / f'audio_worker_{i}.h5') for i in range(n)]

    worker_args = [
        (subsets[i], cosyvoice_root, model_dir, prosody_encoder_path,
         gpu_ids[i], utt2anon_emb_cpu, utt2text, utt2spk, wav_scp,
         utt2alignment, temp_h5s[i], output_sr)
        for i in range(n)
    ]

    with Pool(processes=n) as pool:
        pool.map(_synthesis_worker, worker_args)

    print('  Merging worker HDF5 files...')
    with h5py.File(h5_path, 'a') as hf:
        hf.attrs['cosyvoice2_synthesized'] = True
        for temp_path in temp_h5s:
            if not Path(temp_path).exists():
                print(f'  [warn] worker file missing: {temp_path}')
                continue
            try:
                with h5py.File(temp_path, 'r') as src:
                    n_copied = 0
                    for utt in src:
                        if utt in hf:
                            del hf[utt]
                        src.copy(utt, hf)
                        n_copied += 1
                print(f'  Merged {n_copied} utterances from {Path(temp_path).name}')
            except Exception as e:
                print(f'  [warn] could not read {temp_path}: {e} — worker may have crashed')
            finally:
                try:
                    Path(temp_path).unlink()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpu', default='0',
                        help='Comma-separated GPU indices, e.g. "0,1,2,3"')
    parser.add_argument('--output_sr', type=int, default=16000,
                        help='Output sample rate (default 16000 to match ASR/ASV models)')
    parser.add_argument('--force', action='store_true',
                        help='Re-synthesise even if output already exists')
    parser.add_argument('--asr_text_dir', default=None,
                        help='Directory of ASR text files (e.g. exp/asr). '
                             'All text files found recursively are merged into a '
                             'utt->transcript lookup that overrides ground-truth transcripts.')
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpu.split(',') if g.strip()]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    cfg = load_config(args.config)
    m   = cfg['modules']

    cosyvoice_root = str(m['cosyvoice_root'])
    for p in [cosyvoice_root, os.path.join(cosyvoice_root, 'third_party', 'Matcha-TTS')]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from cosyvoice.cli.cosyvoice import CosyVoice2

    model_dir            = str(m['model_dir'])
    prosody_encoder_path = str(m.get('prosody_encoder_path') or '')
    spk_pool_path        = str(m['spk_pool_path'])
    n_anon               = int(m.get('n_anon_speakers', 10))
    anon_suffix          = str(m['anon_suffix'])
    anon_percentile      = float(m['anon_percentile']) if m.get('anon_percentile') is not None else None
    anon_method          = str(m.get('anon_method', 'farthest'))
    seed                 = int(m.get('seed', 0))
    gender_mode          = str(m.get('gender_mode', 'all'))
    if gender_mode not in ('all', 'same', 'opposite'):
        raise ValueError(f"config gender must be 'all', 'same', or 'opposite', got {gender_mode!r}")

    print(f'Loading model from {model_dir}')
    tts = CosyVoice2(model_dir, load_jit=False, load_trt=False,
                     prosody_encoder_path=prosody_encoder_path or '')
    print(f'Model device: {tts.model.device}, sample_rate: {tts.sample_rate}, GPUs: {gpu_ids}')

    print(f'Loading speaker pool from {spk_pool_path}')
    pool_raw, pool_norm, pool_ids = load_pool(spk_pool_path)
    print(f'Pool: {len(pool_raw)} speakers, norm range '
          f'{pool_raw.norm(dim=1).min():.2f}–{pool_raw.norm(dim=1).max():.2f}')

    pool_genders = {}
    if gender_mode != 'all':
        pool_gender_path = m.get('spk_pool_gender_path')
        if pool_gender_path:
            pool_genders = load_gender_file(str(pool_gender_path))
            print(f'Pool genders loaded: {len(pool_genders)} speakers from {pool_gender_path}')
        else:
            print('Warning: --gender != all but spk_pool_gender_path not set in config')

    if anon_method == 'random':
        print(f'Anon strategy: random (gender={gender_mode}, seed={seed})')
    elif anon_percentile is not None:
        print(f'Anon strategy: anchor at {anon_percentile}th percentile of distance, '
              f'then {n_anon} closest to anchor (gender={gender_mode})')
    else:
        print(f'Anon strategy: mean of {n_anon} farthest from source (gender={gender_mode})')

    asr_transcripts = None
    if args.asr_text_dir:
        print(f'\nLoading ASR transcripts from {args.asr_text_dir}')
        asr_transcripts = load_asr_transcripts(args.asr_text_dir)

    datasets = expand_datasets(cfg)
    print(f'\nDatasets to anonymise ({len(datasets)}):')
    for name, path in datasets:
        print(f'  {name}: {path} -> {str(path) + anon_suffix}')

    for dataset_name, src_path in datasets:
        output_path = Path(str(src_path) + anon_suffix)
        synthesise_dataset(
            dataset_name, src_path, output_path,
            tts, pool_raw, pool_norm, pool_ids, n_anon,
            cosyvoice_root, model_dir, prosody_encoder_path,
            args.output_sr, gpu_ids, args.force,
            asr_transcripts=asr_transcripts,
            anon_percentile=anon_percentile,
            anon_method=anon_method,
            gender_mode=gender_mode,
            pool_genders=pool_genders,
            seed=seed,
        )

    print('\nDone.')


if __name__ == '__main__':
    main()
