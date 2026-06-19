from pathlib import Path
from collections import defaultdict
from shutil import copy
from utils import save_kaldi_format, create_clean_dir, read_kaldi_format, get_datasets
from .logger import setup_logger
import os,sys

logger = setup_logger(__name__)


def combine_asr_data(input_dirs, output_dir):
    output_dir = Path(output_dir)
    create_clean_dir(output_dir)

    files = ['text', 'utt2spk', 'spk2gender', 'wav.scp']

    for file in files:
        data = {}
        for in_dir in input_dirs:
            in_dir = Path(in_dir)
            data.update(read_kaldi_format(in_dir / file))
        save_kaldi_format(data, output_dir / file)

    spk2utt = defaultdict(list)
    for in_dir in input_dirs:
        in_dir = Path(in_dir)
        for spk, utt_list in read_kaldi_format(in_dir / 'spk2utt').items():
            spk2utt[spk].extend(utt_list)
    spk2utt = {spk: sorted(utt_list) for spk, utt_list in spk2utt.items()}
    save_kaldi_format(spk2utt, output_dir / 'spk2utt')


def list_wav_files_recursively(folder_path, extension=".wav"):
    wav_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                wav_files.append(os.path.join(root, file))
    key_to_wav_files = {}
    for wav in wav_files:
        key, _ = os.path.splitext(os.path.basename(wav))
        key_to_wav_files[key] = wav
    return wav_files, key_to_wav_files


def _is_valid_audio_path(audio) -> bool:
    """Return True for regular file paths and for hdf5: URIs."""
    if isinstance(audio, bytes):
        audio = audio.decode('utf-8')
    if audio.startswith('hdf5:'):
        # hdf5:/path/to/file.h5:utt_id — check that the h5 file itself exists
        rest = audio[len('hdf5:'):]
        last_colon = rest.rfind(':')
        h5_file = rest[:last_colon] if last_colon != -1 else rest
        return os.path.isfile(h5_file)
    return os.path.isfile(audio)


def _wav_scp_uses_hdf5(path) -> bool:
    """Return True if the first entry in a wav.scp uses an hdf5: URI."""
    try:
        with open(path, 'r') as f:
            first = f.readline().strip()
        parts = first.split(None, 1)
        return len(parts) == 2 and parts[1].startswith('hdf5:')
    except Exception:
        return False


def check_files(ori_folder, anon_folder, required_files):
    for required_file in required_files:
        if 'wav.scp' in str(required_file):
            continue
        if not os.path.exists(anon_folder / required_file):
            logger.debug(f"{ori_folder / required_file} missing in {anon_folder}, copying the content of {ori_folder / required_file}.")
            ori_file = ori_folder / required_file
            anon_file = anon_folder / required_file
            copy(ori_file, anon_file)

    # wav.scp special case
    if 'wav.scp' in required_files and os.path.exists(anon_folder / 'wav.scp'):
        logger.debug(f"Directory {anon_folder} doesn't contain any audios, if the audios are stored elsewhere and correctly put in wav.scp this is fine.")
        # checking if wav.scp file is coorect (if not, try to create a valid wav.scp with anon_folder/wav/*.wav)
        with open(ori_folder / 'wav.scp', 'rb') as f1, open(anon_folder / 'wav.scp', 'rb') as f2:
            content1 = f1.readlines()
            content2 = f2.readlines()
            if len(content1) != len(content2):
                logger.warning(f"len({ori_folder}/wav.scp) != len({anon_folder}/wav.scp), {len(content1)} != {len(content2)}.")
                a = set([c.strip().split()[0] for c in content1])
                b = set([c.strip().split()[0] for c in content2])
                logger.warning(f"Missing keys: {a-b}")
                # If the anon wav.scp already uses hdf5: URIs, trust it as-is
                if _wav_scp_uses_hdf5(anon_folder / 'wav.scp'):
                    logger.warning(f"wav.scp uses hdf5 URIs — keeping existing wav.scp with {len(content2)} entries.")
                    return
                create_wavscp_formart_data(ori_folder, anon_folder)
                return
            ori_keys = [k.strip().split()[0] for k in content1]
            for a in content2:
                a_key = a.strip().split()[0]
                if a_key not in ori_keys:
                    logger.warning(f"{a_key} duplicated or not in {ori_folder}")
                    create_wavscp_formart_data(ori_folder, anon_folder)
                    return
                ori_keys.remove(a_key)

                audio = a.strip().split()[1]
                if not _is_valid_audio_path(audio):
                    logger.warning(f"{anon_folder / 'wav.scp'}: {a_key} (file: {audio}) is not a valid path.")
                    create_wavscp_formart_data(ori_folder, anon_folder)
                    return
    else: # if no wav.scp create wav.scp
            create_wavscp_formart_data(ori_folder, anon_folder)
    return


def create_wavscp_formart_data(ori_folder, anon_folder):
    logger.info(f"Automatic wav.scp creation for {anon_folder} from {anon_folder}/**/*.wav")
    ori_file = ori_folder / "wav.scp"
    anon_file = anon_folder / "wav.scp"
    wav_scp_content = ""
    with open(ori_file, 'rb') as f2:
        wav_files, key_to_wav_files = list_wav_files_recursively(anon_folder)
        if len(wav_files) == 0:
            logger.error(f"Directory {anon_folder}/**/* doesn't contain any audios, please create a wav.scp file with correct paths, or place the wavs in {anon_folder}/wav/$UTTID.wav")
            exit(1)
        for line in open(ori_file):
            key = line.strip().split(' ')[0]
            if key not in key_to_wav_files:
                logger.error(f"{Path(wav_files[0]).parent} is missing {key}.")
                exit(1)
            wav_scp_content += f"{key} {key_to_wav_files[key]}\n"
    with open(anon_file, 'w') as fp:
        fp.write(wav_scp_content)


def check_kaldi_formart_data(config):
    logger.info('Check data directories format..')
    # Only check that anonymized dataset dirs from config exist: dev$suffix, test$suffix, train-clean-360$suffix, etc.
    dataset_dict = get_datasets(config)
    output_path = config['data_dir']
    suffix = config['anon_data_suffix']

    if 'train_data_name' in config:
        # in conf/eval_post.yaml, train_data_name = train-clean-360$suffix, ori_train_data_name=train-clean-360
        ori_train_data_name = config['train_data_name'].split(suffix)[0]
        dataset_dict[ori_train_data_name] = Path(config['data_dir'], ori_train_data_name)

    for dataset, orig_dataset_path in dataset_dict.items():
        out_data_split = Path(output_path) / f'{dataset}{suffix}'
        if not os.path.exists(out_data_split):
            logger.error(f"Directory {out_data_split} does not exist. Please save your anonymized audio there.")
            exit(1)
        # Generate/copy Kaldi format files for anon dir (utt2spk, text, wav.scp, etc.) from original or anon wavs
        ori_folder = Path(output_path) / dataset
        if ori_folder.is_dir():
            required_files = [f for f in os.listdir(ori_folder) if os.path.isfile(ori_folder / f)]
            check_files(ori_folder, out_data_split, required_files)
        else:
            logger.warning(f"Original folder {ori_folder} does not exist; cannot copy/generate Kaldi files for {out_data_split}.")
    logger.info(f"Checked and prepared Kaldi files for dataset_dict: {list(dataset_dict.keys())}.")
