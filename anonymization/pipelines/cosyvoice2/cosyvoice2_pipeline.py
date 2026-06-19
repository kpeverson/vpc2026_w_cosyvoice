import sys
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from .. import Pipeline, get_anon_level_from_config
from utils import read_kaldi_format, copy_data_dir, setup_logger, load_audio

logger = setup_logger(__name__)


def _wav_path_for_onnx(wav_entry) -> str:
    """
    Return a plain file path usable by torchaudio.load / ONNX tools.
    For hdf5: URIs, writes a temp wav file and returns its path.
    Caller is responsible for deleting the temp file if one is created.
    Returns (path_str, is_temp).
    """
    entry = str(wav_entry[-1] if isinstance(wav_entry, list) else wav_entry)
    if not entry.startswith('hdf5:'):
        return entry, False
    signal, sr = load_audio(entry)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp = f.name
    torchaudio.save(tmp, signal, sr)
    return tmp, True


def _synthesis_worker(args):
    """Worker: one subprocess per GPU. Synthesises a subset of utterances."""
    (utt_subset, cosyvoice_root, model_dir, prosody_encoder_path, gpu_index,
     spk2anon_emb_cpu, utt2text, utt2spk, src_wav_scp,
     temp_h5_path, sample_rate) = args

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
    device_str = f'cuda:{gpu_index}'

    for p in [cosyvoice_root, os.path.join(cosyvoice_root, 'third_party', 'Matcha-TTS')]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from cosyvoice.cli.cosyvoice import CosyVoice2
    tts = CosyVoice2(model_dir, prosody_encoder_path=prosody_encoder_path)
    model_device = tts.model.device
    use_prosody = bool(prosody_encoder_path)

    spk2anon_emb = {spk: emb.to(model_device).unsqueeze(0)
                    for spk, emb in spk2anon_emb_cpu.items()}

    with h5py.File(temp_h5_path, 'w') as h5f:
        for utt in tqdm(utt_subset, desc=f'GPU {gpu_index}', position=gpu_index):
            spk = utt2spk.get(utt)
            text = utt2text.get(utt, '')
            if not text or spk not in spk2anon_emb:
                continue
            try:
                prosody_emb = None
                if use_prosody and utt in src_wav_scp:
                    src_path, is_temp = _wav_path_for_onnx(src_wav_scp[utt])
                    try:
                        prosody_emb = tts._wav_to_prosody_emb(src_path)
                    finally:
                        if is_temp:
                            os.unlink(src_path)

                audio = _run_tts(tts, text, spk2anon_emb[spk], prosody_emb)
                ds = h5f.create_dataset(utt, data=audio)
                ds.attrs['sample_rate'] = sample_rate
            except Exception as e:
                print(f'[GPU {gpu_index}] Synthesis failed for {utt}: {e}')

    return temp_h5_path


def _run_tts(tts, text: str, emb: torch.Tensor,
             prosody_emb=None) -> np.ndarray:
    """Core synthesis: text + speaker emb + optional prosody emb -> float32 numpy array."""
    # CosyVoice2 tokenizer treats uppercase words as acronyms (spells letters individually).
    # IEMOCAP transcripts are all-caps, so lowercase before normalizing.
    text = text.lower()
    chunks = []
    for segment in tts.frontend.text_normalize(text, split=True, text_frontend=True):
        tts_text_token, tts_text_token_len = tts.frontend._extract_text_token(segment)
        model_input = {
            'text': tts_text_token,
            'text_len': tts_text_token_len,
            'llm_embedding': emb,
            'flow_embedding': emb,
        }
        if prosody_emb is not None:
            model_input['prosody_emb'] = prosody_emb
        for out in tts.model.tts(**model_input, stream=False, speed=1.0):
            chunks.append(out['tts_speech'].cpu())
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return torch.cat(chunks, dim=1).squeeze(0).numpy().astype(np.float32)


class CosyVoice2Pipeline(Pipeline):

    def __init__(self, config: dict, force_compute: bool = False, devices: list = None):
        self.config = config
        self.force_compute = force_compute
        self.devices = devices or [torch.device('cuda' if torch.cuda.is_available() else 'cpu')]
        m = config['modules']

        self.cosyvoice_root = str(Path(m['cosyvoice_root']))
        self.model_dir = str(m['model_dir'])
        self.prosody_encoder_path = str(m.get('prosody_encoder_path', '') or '')

        for p in [self.cosyvoice_root,
                  os.path.join(self.cosyvoice_root, 'third_party', 'Matcha-TTS')]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from cosyvoice.cli.cosyvoice import CosyVoice2
        logger.info(f'Loading CosyVoice2 from {self.model_dir}')
        if self.prosody_encoder_path:
            logger.info(f'Prosody encoder: {self.prosody_encoder_path}')
        self.tts = CosyVoice2(self.model_dir,
                              prosody_encoder_path=self.prosody_encoder_path)
        self.sample_rate = self.tts.sample_rate
        logger.info(f'Model device: {self.tts.model.device}, n_gpus: {len(self.devices)}')

        logger.info(f'Loading speaker pool from {m["spk_pool_path"]}')
        raw = torch.load(m['spk_pool_path'], map_location='cpu')
        spk_ids = list(raw.keys())
        spk_matrix = torch.tensor([raw[k] for k in spk_ids], dtype=torch.float32)
        self.pool_ids = spk_ids
        self.pool_matrix_raw = spk_matrix                   # original scale, passed to model
        self.pool_matrix = F.normalize(spk_matrix, dim=1)  # unit-norm, for cosine sim only
        self.n_anon = int(m.get('n_anon_speakers', 10))

    # ------------------------------------------------------------------
    # Speaker embedding helpers
    # ------------------------------------------------------------------

    def _extract_embedding(self, wav_path) -> torch.Tensor:
        path, is_temp = _wav_path_for_onnx(wav_path)
        try:
            emb = self.tts.frontend._extract_spk_embedding(path)
        finally:
            if is_temp:
                os.unlink(path)
        return emb.squeeze(0)

    def _build_anon_embedding(self, source_emb: torch.Tensor) -> torch.Tensor:
        src = F.normalize(source_emb.unsqueeze(0), dim=1)
        sims = (src @ self.pool_matrix.T).squeeze(0)
        _, indices = torch.topk(sims, self.n_anon, largest=False)
        # Return raw-scale mean — model was trained with unnormalized CAMPlus embeddings (~norm 10)
        return self.pool_matrix_raw[indices].mean(dim=0)

    def _load_speaker_embeddings(self, dataset_path: Path, wav_scp: dict, utt2spk: dict) -> dict:
        spk2emb_path = dataset_path / 'spk2embedding.pt'
        if spk2emb_path.exists():
            logger.info(f'Loading pre-extracted embeddings from {spk2emb_path}')
            raw = torch.load(spk2emb_path, map_location='cpu')
            return {spk: torch.tensor(v, dtype=torch.float32) for spk, v in raw.items()}

        logger.warning(
            f'No spk2embedding.pt in {dataset_path} — extracting on the fly. '
            'Run extract_cosyvoice_embeddings.py offline for faster future runs.'
        )
        spk2embs: dict = {}
        for utt, wav_path in tqdm(wav_scp.items(), desc='Extracting embeddings'):
            spk = utt2spk.get(utt)
            if spk is None:
                continue
            try:
                emb = self._extract_embedding(wav_path)
                spk2embs.setdefault(spk, []).append(emb)
            except Exception as e:
                logger.warning(f'Embedding extraction failed for {utt}: {e}')
        return {spk: torch.stack(embs).mean(0) for spk, embs in spk2embs.items()}

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run_anonymization_pipeline(self, datasets: dict):
        for i, (dataset_name, dataset_path) in enumerate(datasets.items()):
            anon_level = get_anon_level_from_config(self.config['modules'], dataset_name)
            logger.info(f'{i+1}/{len(datasets)}: CosyVoice2 anonymizing "{dataset_name}" '
                        f'(anon_level={anon_level}, prosody={bool(self.prosody_encoder_path)})')

            output_path = Path(str(dataset_path) + self.config['modules']['anon_suffix'])

            # Check for completed synthesis BEFORE copy_data_dir, which would overwrite audio.h5
            h5_path = output_path / 'audio.h5'
            if not self.force_compute and h5_path.exists():
                with h5py.File(h5_path, 'r') as _f:
                    if bool(_f.attrs.get('cosyvoice2_synthesized', False)):
                        logger.info(f'Already synthesised {dataset_name}, skipping.')
                        continue

            copy_data_dir(dataset_path, output_path)

            # copy_data_dir may have copied the source audio.h5 — delete it if not synthesized
            if h5_path.exists():
                with h5py.File(h5_path, 'r') as _f:
                    is_synthesized = bool(_f.attrs.get('cosyvoice2_synthesized', False))
                if not is_synthesized or self.force_compute:
                    h5_path.unlink()

            src_wav_scp = read_kaldi_format(dataset_path / 'wav.scp')
            utt2spk = read_kaldi_format(dataset_path / 'utt2spk')
            text_file = dataset_path / 'text'
            if not text_file.exists():
                raise FileNotFoundError(f'No text file at {text_file}')
            utt2text = read_kaldi_format(text_file, values_as_string=True)

            spk2src_emb = self._load_speaker_embeddings(dataset_path, src_wav_scp, utt2spk)
            spk2anon_emb = {spk: self._build_anon_embedding(emb)
                            for spk, emb in spk2src_emb.items()}

            already_done = set()
            if h5_path.exists() and not self.force_compute:
                with h5py.File(h5_path, 'r') as _f:
                    already_done = set(_f.keys())

            pending = [utt for utt in src_wav_scp
                       if utt not in already_done
                       and utt2spk.get(utt) in spk2anon_emb
                       and utt2text.get(utt, '')]

            if not pending:
                logger.info(f'All utterances already synthesised for {dataset_name}')
            elif len(self.devices) == 1:
                self._synthesise_single_gpu(pending, h5_path, spk2anon_emb,
                                            utt2text, utt2spk, src_wav_scp)
            else:
                self._synthesise_multi_gpu(pending, output_path, h5_path, spk2anon_emb,
                                           utt2text, utt2spk, src_wav_scp)

            with h5py.File(h5_path, 'r') as h5f:
                scp_lines = [f'{utt} hdf5:{h5_path.resolve()}:{utt}\n'
                             for utt in sorted(h5f.keys())]
            with open(output_path / 'wav.scp', 'w') as f:
                f.writelines(scp_lines)

        logger.info('Done.')

    def _synthesise_single_gpu(self, pending, h5_path, spk2anon_emb,
                               utt2text, utt2spk, src_wav_scp):
        model_device = self.tts.model.device
        emb_on_device = {spk: emb.to(model_device).unsqueeze(0)
                         for spk, emb in spk2anon_emb.items()}

        with h5py.File(h5_path, 'a') as h5f:
            h5f.attrs['cosyvoice2_synthesized'] = True
            for utt in tqdm(pending, desc='Synthesising'):
                spk = utt2spk[utt]
                text = utt2text[utt]
                try:
                    prosody_emb = None
                    if self.prosody_encoder_path and utt in src_wav_scp:
                        src_path, is_temp = _wav_path_for_onnx(src_wav_scp[utt])
                        try:
                            prosody_emb = self.tts._wav_to_prosody_emb(src_path)
                        finally:
                            if is_temp:
                                os.unlink(src_path)

                    audio = _run_tts(self.tts, text, emb_on_device[spk], prosody_emb)
                    if utt in h5f:
                        del h5f[utt]
                    ds = h5f.create_dataset(utt, data=audio)
                    ds.attrs['sample_rate'] = self.sample_rate
                except Exception as e:
                    logger.warning(f'Synthesis failed for {utt}: {e}')

    def _synthesise_multi_gpu(self, pending, output_path, h5_path, spk2anon_emb,
                              utt2text, utt2spk, src_wav_scp):
        from torch.multiprocessing import Pool, set_start_method
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        n = len(self.devices)
        subsets = [pending[i::n] for i in range(n)]
        temp_h5s = [str(output_path / f'audio_worker_{i}.h5') for i in range(n)]

        worker_args = [
            (subsets[i], self.cosyvoice_root, self.model_dir, self.prosody_encoder_path,
             self.devices[i].index, spk2anon_emb, utt2text, utt2spk, src_wav_scp,
             temp_h5s[i], self.sample_rate)
            for i in range(n)
        ]

        with Pool(processes=n) as pool:
            pool.map(_synthesis_worker, worker_args)

        logger.info('Merging worker HDF5 files...')
        with h5py.File(h5_path, 'a') as h5f:
            h5f.attrs['cosyvoice2_synthesized'] = True
            for temp_path in temp_h5s:
                if not Path(temp_path).exists():
                    continue
                with h5py.File(temp_path, 'r') as src:
                    for utt in src:
                        if utt in h5f:
                            del h5f[utt]
                        src.copy(utt, h5f)
                Path(temp_path).unlink()
