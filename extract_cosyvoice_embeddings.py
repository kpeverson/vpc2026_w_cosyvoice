"""
Extract CAMPlus speaker embeddings for Kaldi-format data directories whose
wav.scp may contain hdf5: URIs.  Saves utt2embedding.pt and spk2embedding.pt
in the same list-of-floats format used by the GigaSpeech CosyVoice2 data.

Usage:
    python extract_cosyvoice_embeddings.py \\
        --onnx_path /gscratch/tial/kpever/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B/campplus.onnx \\
        --dirs data/libri_dev_enrolls data/libri_dev_trials_mixed \\
               data/libri_test_enrolls data/libri_test_trials_mixed \\
               data/IEMOCAP_dev data/IEMOCAP_test \\
        --num_thread 8
"""
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm

# Thread-local HDF5 handles (one open file descriptor per thread per file)
_thread_local = threading.local()
_h5_lock = threading.Lock()
_h5_paths: set[str] = set()


def _get_h5(path: str):
    if not hasattr(_thread_local, 'handles'):
        _thread_local.handles = {}
    if path not in _thread_local.handles:
        _thread_local.handles[path] = h5py.File(path, 'r')
    return _thread_local.handles[path]


def _load_wav(wav_entry) -> tuple[torch.Tensor, int]:
    """Load audio from a plain path or hdf5:/path/to/file.h5:key URI."""
    if isinstance(wav_entry, list):
        wav_entry = wav_entry[-1]
    wav_entry = str(wav_entry)
    if wav_entry.startswith('hdf5:'):
        rest = wav_entry[len('hdf5:'):]
        last_colon = rest.rfind(':')
        h5_path, key = rest[:last_colon], rest[last_colon + 1:]
        f = _get_h5(h5_path)
        arr = np.array(f[key][:], dtype=np.float32)
        audio = torch.from_numpy(arr)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        sr = int(f[key].attrs.get('sample_rate', 16000))
        return audio, sr
    return torchaudio.load(wav_entry)


def _extract_embedding(wav_entry, ort_session) -> list[float] | None:
    try:
        audio, sr = _load_wav(wav_entry)
    except Exception as e:
        return None
    if sr != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)
    feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = ort_session.run(
        None, {ort_session.get_inputs()[0].name: feat.unsqueeze(0).numpy()}
    )[0].flatten().tolist()
    return embedding


def process_dir(data_dir: Path, ort_session, num_thread: int, force: bool):
    utt2emb_path = data_dir / 'utt2embedding.pt'
    spk2emb_path = data_dir / 'spk2embedding.pt'
    if utt2emb_path.exists() and spk2emb_path.exists() and not force:
        print(f'Skipping {data_dir}: embeddings already exist')
        return

    wav_scp_path = data_dir / 'wav.scp'
    utt2spk_path = data_dir / 'utt2spk'
    if not wav_scp_path.exists() or not utt2spk_path.exists():
        print(f'Skipping {data_dir}: missing wav.scp or utt2spk')
        return

    utt2wav = {}
    with open(wav_scp_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                utt2wav[parts[0]] = parts[1]

    utt2spk = {}
    with open(utt2spk_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                utt2spk[parts[0]] = parts[1]

    print(f'Processing {data_dir}: {len(utt2wav)} utterances')

    def job(utt):
        emb = _extract_embedding(utt2wav[utt], ort_session)
        return utt, emb

    utt2embedding, spk2embedding = {}, {}
    skipped = 0
    with ThreadPoolExecutor(max_workers=num_thread) as executor:
        futures = {executor.submit(job, u): u for u in utt2wav}
        for future in tqdm(as_completed(futures), total=len(futures), desc=str(data_dir)):
            utt, emb = future.result()
            if emb is None:
                skipped += 1
                continue
            utt2embedding[utt] = emb
            spk = utt2spk.get(utt)
            if spk:
                spk2embedding.setdefault(spk, []).append(emb)

    if skipped:
        print(f'  Warning: {skipped} utterances skipped')

    # Average utterance embeddings per speaker (same as GigaSpeech format)
    spk2embedding = {
        spk: torch.tensor(embs).mean(dim=0).tolist()
        for spk, embs in spk2embedding.items()
    }

    torch.save(utt2embedding, utt2emb_path)
    torch.save(spk2embedding, spk2emb_path)
    print(f'  Saved {len(utt2embedding)} utt embeddings, {len(spk2embedding)} spk embeddings -> {data_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', required=True)
    parser.add_argument('--dirs', nargs='+', required=True)
    parser.add_argument('--num_thread', type=int, default=8)
    parser.add_argument('--force', action='store_true', help='Recompute even if files exist')
    args = parser.parse_args()

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    ort_session = onnxruntime.InferenceSession(
        args.onnx_path, sess_options=option, providers=['CPUExecutionProvider']
    )

    for d in args.dirs:
        process_dir(Path(d), ort_session, args.num_thread, args.force)


if __name__ == '__main__':
    main()
