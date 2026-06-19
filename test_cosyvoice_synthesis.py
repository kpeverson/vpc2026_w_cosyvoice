"""
Quick sanity-check: synthesise N utterances from IEMOCAP_dev using the
CosyVoice2 anonymization pipeline and save them as wavs for listening.

Usage:
    python test_cosyvoice_synthesis.py [--n 5] [--out_dir /tmp/cosyvoice_test]
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=5, help='Number of utterances to synthesise')
parser.add_argument('--config', default='configs/track1/anon_cosyvoice2.yaml')
parser.add_argument('--out_dir', default='/tmp/cosyvoice_test')
parser.add_argument('--gpu', default='0')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from utils import parse_yaml, read_kaldi_format, setup_logger
import torch.nn.functional as F

logger = setup_logger('test_synthesis')

config = parse_yaml(args.config)
m = config['modules']

cosyvoice_root = str(m['cosyvoice_root'])
for p in [cosyvoice_root, os.path.join(cosyvoice_root, 'third_party', 'Matcha-TTS')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cosyvoice.cli.cosyvoice import CosyVoice2

logger.info(f'Loading model from {m["model_dir"]}')
tts = CosyVoice2(str(m['model_dir']))
sample_rate = tts.sample_rate
logger.info(f'Model sample_rate: {sample_rate}, device: {tts.model.device}')

# Load speaker pool (raw scale — fix for norm issue)
logger.info(f'Loading speaker pool from {m["spk_pool_path"]}')
pool_raw_dict = torch.load(m['spk_pool_path'], map_location='cpu')
pool_ids = list(pool_raw_dict.keys())
pool_matrix_raw = torch.tensor([pool_raw_dict[k] for k in pool_ids], dtype=torch.float32)
pool_matrix_norm = F.normalize(pool_matrix_raw, dim=1)
n_anon = int(m.get('n_anon_speakers', 10))
logger.info(f'Pool size: {len(pool_ids)}, embedding norm range: '
            f'{pool_matrix_raw.norm(dim=1).min():.2f}–{pool_matrix_raw.norm(dim=1).max():.2f}')

data_dir = Path('data/IEMOCAP_dev')
spk2emb_path = data_dir / 'spk2embedding.pt'
utt2spk = read_kaldi_format(data_dir / 'utt2spk')
utt2text = read_kaldi_format(data_dir / 'text', values_as_string=True)

if spk2emb_path.exists():
    logger.info(f'Loading pre-extracted embeddings from {spk2emb_path}')
    raw_emb = torch.load(spk2emb_path, map_location='cpu')
    spk2src_emb = {spk: torch.tensor(v, dtype=torch.float32) for spk, v in raw_emb.items()}
else:
    raise FileNotFoundError(f'No spk2embedding.pt at {spk2emb_path} — run extract_cosyvoice_embeddings.py first')

logger.info(f'Source embedding norm sample: {list(spk2src_emb.values())[0].norm():.2f}')

def build_anon_emb(src_emb):
    src_norm = F.normalize(src_emb.unsqueeze(0), dim=1)
    sims = (src_norm @ pool_matrix_norm.T).squeeze(0)
    _, idx = torch.topk(sims, n_anon, largest=False)
    anon = pool_matrix_raw[idx].mean(dim=0)
    logger.info(f'  anon embedding norm: {anon.norm():.2f}')
    return anon

# Pick N utterances with non-empty text
candidates = [(utt, utt2text[utt]) for utt in utt2text
              if utt2text.get(utt, '').strip() and utt2spk.get(utt) in spk2src_emb]
sample_utts = candidates[:args.n]

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

model_device = tts.model.device

for utt, text in sample_utts:
    spk = utt2spk[utt]
    anon_emb = build_anon_emb(spk2src_emb[spk]).to(model_device).unsqueeze(0)

    text_lower = text.lower()
    logger.info(f'Synthesising {utt}: "{text_lower}"')
    chunks = []
    for seg in tts.frontend.text_normalize(text_lower, split=True, text_frontend=True):
        tts_text_token, tts_text_token_len = tts.frontend._extract_text_token(seg)
        model_input = {
            'text': tts_text_token,
            'text_len': tts_text_token_len,
            'llm_embedding': anon_emb,
            'flow_embedding': anon_emb,
        }
        for out in tts.model.tts(**model_input, stream=False, speed=1.0):
            chunks.append(out['tts_speech'].cpu())

    if not chunks:
        logger.warning(f'  No output for {utt}')
        continue

    audio = torch.cat(chunks, dim=1)  # [1, T]
    dur = audio.shape[1] / sample_rate
    logger.info(f'  Output: {audio.shape[1]} samples at {sample_rate} Hz = {dur:.2f}s')

    wav_path = out_dir / f'{utt}.wav'
    torchaudio.save(str(wav_path), audio, sample_rate)
    logger.info(f'  Saved to {wav_path}')

logger.info(f'Done. Files in {out_dir}')
