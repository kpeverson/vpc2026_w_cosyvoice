"""
Pack all wav files in a Kaldi data directory into a single HDF5 file and rewrite wav.scp.

Usage:
    python pack_wav_to_hdf5.py data/libri_dev
    python pack_wav_to_hdf5.py data/libri_dev data/libri_test data/IEMOCAP_dev data/IEMOCAP_test

The original wav files are NOT deleted. To reclaim inodes, delete data/<dir>/wav/ after verifying.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torchaudio
from tqdm import tqdm


def pack_directory(data_dir: Path):
    wav_scp = data_dir / 'wav.scp'
    if not wav_scp.exists():
        print(f'Skipping {data_dir}: no wav.scp found')
        return

    entries = {}
    with open(wav_scp) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                entries[parts[0]] = parts[1]

    if not entries:
        print(f'Skipping {data_dir}: wav.scp is empty')
        return

    # Check if already packed
    first_val = next(iter(entries.values()))
    if first_val.startswith('hdf5:'):
        print(f'Skipping {data_dir}: already packed')
        return

    h5_path = data_dir / 'audio.h5'
    new_scp_lines = []

    print(f'Packing {len(entries)} utterances from {data_dir} -> {h5_path}')
    with h5py.File(h5_path, 'w') as h5f:
        for utt_id, wav_path in tqdm(entries.items(), desc=str(data_dir)):
            try:
                signal, sr = torchaudio.load(wav_path)
            except Exception as e:
                print(f'Warning: could not load {wav_path}: {e}', file=sys.stderr)
                continue
            audio = signal.squeeze(0).numpy().astype(np.float32)
            ds = h5f.create_dataset(utt_id, data=audio)
            ds.attrs['sample_rate'] = sr
            new_scp_lines.append(f'{utt_id} hdf5:{h5_path.resolve()}:{utt_id}\n')

    # Rewrite wav.scp
    with open(wav_scp, 'w') as f:
        f.writelines(sorted(new_scp_lines))

    print(f'Done. Rewrote {wav_scp}. Original wav files still exist — delete wav/ to reclaim inodes.')


def main():
    parser = argparse.ArgumentParser(description='Pack wav files into HDF5 and rewrite wav.scp')
    parser.add_argument('dirs', nargs='+', type=Path, help='Kaldi data directories to pack')
    args = parser.parse_args()

    for d in args.dirs:
        if not d.is_dir():
            print(f'Skipping {d}: not a directory', file=sys.stderr)
            continue
        pack_directory(d)


if __name__ == '__main__':
    main()
