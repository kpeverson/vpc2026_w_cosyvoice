"""
Standalone CosyVoice2 anonymization — runs in the 'cosyvoice' conda env.

Replicates CosyVoice2Pipeline without importing the VPC framework, so it
works with torch 2.3.x (cosyvoice env) rather than the venv's torch 2.8.x.

Usage:
    conda run -n cosyvoice -- python run_cosyvoice_anon_standalone.py \
        --config configs/track1/anon_cosyvoice2.yaml [--gpu 0] [--force]
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

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


def load_asr_transcripts(asr_text_dir: str) -> dict:
    """Load all Kaldi text files under asr_text_dir into a merged utt->transcript dict."""
    merged = {}
    root = Path(asr_text_dir)
    for text_file in root.rglob('text'):
        entries = read_kaldi(text_file, values_as_string=True)
        merged.update(entries)
    print(f'  Loaded {len(merged)} ASR transcripts from {asr_text_dir}')
    return merged


def load_speaker_embeddings(dataset_path, wav_scp, utt2spk, tts):
    """Load or extract per-speaker embeddings. Returns {spk: tensor [192]}."""
    pt_path = dataset_path / 'spk2embedding.pt'
    if pt_path.exists():
        print(f'  Loading embeddings from {pt_path}')
        raw = torch.load(pt_path, map_location='cpu')
        return {spk: torch.tensor(v, dtype=torch.float32) for spk, v in raw.items()}

    print(f'  No spk2embedding.pt — extracting on-the-fly (slow)')
    spk2embs = {}
    for utt, wav_path in tqdm(wav_scp.items(), desc='  Extracting embeddings'):
        spk = utt2spk.get(utt)
        if spk is None:
            continue
        try:
            path, is_tmp = write_temp_wav(wav_path)
            try:
                emb = tts.frontend._extract_spk_embedding(path).squeeze(0).cpu()
            finally:
                if is_tmp:
                    os.unlink(path)
            spk2embs.setdefault(spk, []).append(emb)
        except Exception as e:
            print(f'  [warn] embedding failed for {utt}: {e}')
    return {spk: torch.stack(embs).mean(0) for spk, embs in spk2embs.items()}


# ---------------------------------------------------------------------------
# Multi-GPU worker (module-level for pickle compatibility with spawn)
# ---------------------------------------------------------------------------

def _synthesis_worker(args):
    (utt_subset, cosyvoice_root, model_dir, prosody_encoder_path,
     gpu_index, spk2anon_emb_cpu, utt2text, utt2spk, wav_scp,
     temp_h5_path, output_sr) = args

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)

    for p in [cosyvoice_root, os.path.join(cosyvoice_root, 'third_party', 'Matcha-TTS')]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from cosyvoice.cli.cosyvoice import CosyVoice2
    tts = CosyVoice2(model_dir, load_jit=False, load_trt=False,
                     prosody_encoder_path=prosody_encoder_path or '')
    device = tts.model.device
    use_prosody = bool(prosody_encoder_path)

    spk2anon_emb = {spk: emb.to(device) for spk, emb in spk2anon_emb_cpu.items()}

    with h5py.File(temp_h5_path, 'w') as hf:
        for utt in tqdm(utt_subset, desc=f'GPU {gpu_index}', position=gpu_index):
            spk  = utt2spk.get(utt)
            text = utt2text.get(utt, '').lower()
            if not text or spk not in spk2anon_emb:
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

                chunks = []
                for seg in tts.frontend.text_normalize(text, split=True):
                    tok, tok_len = tts.frontend._extract_text_token(seg)
                    model_input = {
                        'text': tok, 'text_len': tok_len,
                        'llm_embedding': spk2anon_emb[spk],
                        'flow_embedding': spk2anon_emb[spk],
                    }
                    if prosody_emb is not None:
                        model_input['prosody_emb'] = prosody_emb
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

    spk2src_emb = load_speaker_embeddings(src_path, wav_scp, utt2spk, tts)

    src_spk2gender = load_gender_file(src_path / 'spk2gender') if gender_mode != 'all' else {}
    if gender_mode != 'all' and not src_spk2gender:
        print(f'  Warning: --gender={gender_mode} but no spk2gender in {src_path}; using all')
    if gender_mode != 'all' and not pool_genders:
        print(f'  Warning: --gender={gender_mode} but spk_pool_gender_path not set; using all')

    rng = np.random.default_rng(seed)

    # Keep anon embeddings on CPU — workers move them to their own device
    spk2anon_emb_cpu = {
        spk: select_anon_embedding(
            emb, spk, src_spk2gender.get(spk), pool_raw, pool_norm, pool_ids,
            pool_genders, n_anon, anon_percentile, anon_method, gender_mode, rng,
        )
        for spk, emb in spk2src_emb.items()
    }

    # Load already-done keys for resumability
    already_done = set()
    if not force and h5_path.exists():
        with h5py.File(h5_path, 'r') as hf:
            already_done = set(hf.keys())

    pending = [
        utt for utt in wav_scp
        if utt not in already_done
        and utt2spk.get(utt) in spk2anon_emb_cpu
        and utt2text.get(utt, '').strip()
    ]

    if not pending:
        print(f'  All {len(already_done)} utterances already done.')
    elif len(gpu_ids) == 1:
        _run_single_gpu(pending, h5_path, tts, spk2anon_emb_cpu, utt2text, utt2spk,
                        wav_scp, prosody_encoder_path, output_sr, dataset_name)
    else:
        _run_multi_gpu(pending, output_path, h5_path, spk2anon_emb_cpu, utt2text, utt2spk,
                       wav_scp, cosyvoice_root, model_dir, prosody_encoder_path,
                       output_sr, gpu_ids)

    # Write wav.scp
    with h5py.File(h5_path, 'r') as hf:
        lines = [f'{utt} hdf5:{h5_path.resolve()}:{utt}\n' for utt in sorted(hf.keys())]
    with open(output_path / 'wav.scp', 'w') as f:
        f.writelines(lines)
    print(f'  Wrote {len(lines)} entries to wav.scp')


def _run_single_gpu(pending, h5_path, tts, spk2anon_emb_cpu, utt2text, utt2spk,
                    wav_scp, prosody_encoder_path, output_sr, desc):
    print(f'  Synthesising {len(pending)} utterances on 1 GPU...')
    device = tts.model.device
    use_prosody = bool(prosody_encoder_path)
    spk2anon_emb = {spk: emb.to(device) for spk, emb in spk2anon_emb_cpu.items()}

    with h5py.File(h5_path, 'a') as hf:
        hf.attrs['cosyvoice2_synthesized'] = True
        for utt in tqdm(pending, desc=f'  {desc}'):
            spk  = utt2spk[utt]
            text = utt2text[utt].lower()
            emb  = spk2anon_emb[spk]
            try:
                prosody_emb = None
                if use_prosody and utt in wav_scp:
                    path, is_tmp = write_temp_wav(wav_scp[utt])
                    try:
                        prosody_emb = tts._wav_to_prosody_emb(path)
                    finally:
                        if is_tmp:
                            os.unlink(path)

                chunks = []
                for seg in tts.frontend.text_normalize(text, split=True):
                    tok, tok_len = tts.frontend._extract_text_token(seg)
                    model_input = {
                        'text': tok, 'text_len': tok_len,
                        'llm_embedding': emb, 'flow_embedding': emb,
                    }
                    if prosody_emb is not None:
                        model_input['prosody_emb'] = prosody_emb
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


def _run_multi_gpu(pending, output_path, h5_path, spk2anon_emb_cpu, utt2text, utt2spk,
                   wav_scp, cosyvoice_root, model_dir, prosody_encoder_path,
                   output_sr, gpu_ids):
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
         gpu_ids[i], spk2anon_emb_cpu, utt2text, utt2spk, wav_scp,
         temp_h5s[i], output_sr)
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
    gender_mode          = str(m.get('gender', 'all'))
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
