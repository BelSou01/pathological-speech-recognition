import librosa
import numpy as np


def load_audio(file_path, sample_rate=16000):
    """
    Load an audio file and resample it to the target sampling rate.

    Parameters
    ----------
    file_path : str
        Path to the audio file.
    sample_rate : int
        Target sampling rate.

    Returns
    -------
    audio : np.ndarray
        Audio waveform.
    sr : int
        Sampling rate.
    """

    audio, sr = librosa.load(
        file_path,
        sr=sample_rate,
        mono=True
    )

    return audio, sr


def normalize_audio(audio):
    """
    Normalize an audio waveform.

    Parameters
    ----------
    audio : np.ndarray
        Audio waveform.

    Returns
    -------
    np.ndarray
        Normalized waveform.
    """

    max_value = np.max(np.abs(audio))

    if max_value == 0:
        return audio

    return audio / max_value
