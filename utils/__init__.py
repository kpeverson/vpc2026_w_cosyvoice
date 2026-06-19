# Apply torchaudio compatibility patch before any speechbrain imports
try:
    import torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        def list_audio_backends():
            return ['soundfile', 'sox', 'ffmpeg']
        torchaudio.list_audio_backends = list_audio_backends
except ImportError:
    pass  # torchaudio may not be installed yet

from .data_io import read_kaldi_format, save_kaldi_format, parse_yaml, save_yaml, write_table, load_wav_from_scp, load_audio
from .path_management import (create_clean_dir, remove_contents_in_dir, get_datasets,
                              scan_checkpoint, copy_data_dir)
from .prepare_results_in_kaldi_format import combine_asr_data,check_kaldi_formart_data
from .dependencies import check_dependencies
from .logger import setup_logger
from .select_utterances import limit_utts_per_speaker