import functools
import librosa
import numpy as np
import os
import pandas as pd
import pathlib


def stft_features(signal, **kwargs):
    stft = librosa.stft(signal, **kwargs)
    stft_db = np.abs(stft)
    return stft_db


class SignalWindowDataset:
    def __init__(self, folder_path, sr=None, crop_len_sec=1) -> None:
        path_to_data = pathlib.Path(folder_path)
        wav_files = list(path_to_data.glob('**/*.wav'))

        self.file_list = wav_files
        self.sr = sr
        self.crop_len_sec = crop_len_sec  # HOw many seconds of the audio to consider in one sample

        # Extract STFT features on disjoint windows!
        window_size = 1024
        self.feature_extractor = functools.partial(stft_features, n_fft=window_size, hop_length=window_size)
        self.window_size = window_size
        
        self._df_stats = None
    
    @property
    def df_stats(self,):
        if self._df_stats is not None:
            return self._df_stats

        durations = []
        for file_path in self.file_list:
            signal, sr = librosa.load(file_path, sr=None)
            durations.append({
                'file_path': file_path,
                'sr': sr,
                'frames': signal.shape[0],
            })
        _df_stats = pd.DataFrame(durations)
        _df_stats['seconds'] = _df_stats.frames / _df_stats.sr
        self._df_stats = _df_stats
        return _df_stats

    def __getitem__(self, ix):
        file_path = self.file_list[ix]

        # If the annotation for this file is missing, raise a ValueError!
        annot_file_path = file_path.with_suffix('.txt')
        target_onsets = self._load_target_onsets(annot_file_path=annot_file_path)

        # Load the audio file
        signal, sr_final = librosa.load(str(file_path), sr=self.sr)

        # Choose a random sample of fixed length
        signal_crop, onsets_in_sample, start_frame_crop = self._get_random_signal_crop(
            signal=signal, sr=sr_final, onsets=target_onsets, crop_len_sec=self.crop_len_sec
        )

        # Extract features for the crop
        features = self.feature_extractor(signal_crop)  # n_feats * n_windows
        n_windows = features.shape[1]

        # For each disjoint window, mark 1 if an onset was present in it, else 0
        labels = []
        for ix in range(n_windows):
            start_frame = max(0, int(ix * self.window_size - self.window_size / 2))
            end_frame = min(signal_crop.shape[0], ix * self.window_size + self.window_size / 2)

            start_s, end_s = start_frame / sr_final, end_frame / sr_final
            # Get the target label
            onsets_in_window = self._get_onsets_in_range(
                onsets=onsets_in_sample, relative=False,
                start_s=start_s, end_s=end_s,
            )
            labels.append(len(onsets_in_window) > 0)  # If there is at least one onset in the window, mark 1, else 0.

        labels = np.array(labels).astype(np.long)
        assert labels.shape[0] == n_windows

        return {
            'signal': signal_crop,
            'sr': sr_final,
            'onsets': onsets_in_sample,

            'features': features[np.newaxis, ...],  # 1 * n_windows * n_feats
            'labels': labels[np.newaxis, ...],  # 1 * n_windows
            'file_path': str(annot_file_path),
            'start_s': start_frame_crop/sr_final,
        }

    @staticmethod
    def _get_onsets_in_range(onsets, start_s, end_s, relative=True):
        onsets_in_range = onsets[np.bitwise_and(start_s <= onsets[:], onsets[:] <= end_s)]
        if relative:
            onsets_in_range -= start_s
        return onsets_in_range

    @staticmethod
    def _get_random_signal_crop(signal, sr, onsets, crop_len_sec):
        n_total_frames = signal.shape[0]
        n_frames_in_crop = crop_len_sec * sr

        start_frame = np.random.randint(0, max(n_total_frames - n_frames_in_crop, 1))
        end_frame = min(start_frame + n_frames_in_crop, n_total_frames)

        signal_sample = signal[start_frame: end_frame]
        onsets_in_sample = SignalWindowDataset._get_onsets_in_range(
            onsets, start_s=(start_frame / sr), end_s=(end_frame / sr), relative=True
        )

        return signal_sample, onsets_in_sample, start_frame

    @staticmethod
    def _load_target_onsets(annot_file_path) -> np.ndarray:
        if not os.path.exists(annot_file_path):
            raise FileNotFoundError(str(annot_file_path))
        df_annot = pd.read_csv(annot_file_path, sep='\t', )
        target_onsets = df_annot.OnsetTime.values

        target_onsets = np.round(target_onsets, decimals=2)
        target_onsets = np.unique(target_onsets)
        return target_onsets

    def __len__(self):
        return len(self.file_list)


if __name__ == '__main__':
    dataset = SignalWindowDataset(
        folder_path='../data',
    )

    datum = dataset[0]
    print(datum['labels'])
