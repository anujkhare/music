from typing import Tuple, List
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


N_NOTES = 88  # FIXME: This is hard-coded for the piano music from MAPS


class SignalWindowDataset:
    """
    1. Randomly crop the audio to a fixed length
    2. Get STFT features using disjoint windows (hop_length = window_length)
    3. For each window, label:
        1. Onsets: 1 if there is at least one onset within the window, 0 otherwise
        2. Notes: ONLY in windows that have onsets, mark 1 for the NEW notes (corresponding to the onsets)
    """
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
    def df_stats(self, ):
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
        _df_stats.loc[:, 'seconds'] = _df_stats.frames / _df_stats.sr
        self._df_stats = _df_stats
        return _df_stats

    @property
    def sampling_weights(self) -> np.ndarray:
        """
        Get the sampling weights to be used for each element.

        Since we only feed a fixed window of 2s from each song, if the shortest song in the dataset is of 2s
        and the longest one is 8s, we should ideally sample the longer song 4x more than the shorter one to ensure
        that we see all the 2s windows roughly the same number of times.

        These weights are very approximate and perfectly balanced sampling requires us to consider the number of windows
        in each song (which depends on crop_len_in_sec) and on the starting points!
        """
        df_stats = self.df_stats
        durations = df_stats['seconds']
        weights = durations / durations.min()
        return weights

    def __getitem__(self, ix):
        file_path = self.file_list[ix]

        # If the annotation for this file is missing, raise a ValueError!
        annot_file_path = file_path.with_suffix('.txt')

        # Load the audio file and the annotations
        signal, sr_final = librosa.load(str(file_path), sr=self.sr)
        df_notes = self._load_annotations_df(str(annot_file_path))

        # Choose a random sample of fixed length
        signal_crop, start_s_crop, end_s_crop = self._get_random_signal_crop(
            signal=signal, sr=sr_final, crop_len_sec=self.crop_len_sec
        )
        df_notes_in_crop = self._get_rows_in_range(
            df_annots=df_notes, start_s=start_s_crop, end_s=end_s_crop, relative=True
        )
        target_onsets = np.unique(df_notes_in_crop['OnsetTime'])

        # Extract features for the crop
        features = self.feature_extractor(signal_crop)  # n_feats * n_windows
        n_windows = features.shape[1]

        # For each disjoint window, mark 1 if an onset was present in it, else 0
        labels, note_vectors = [], []
        for ix in range(n_windows):
            start_frame = max(0, int(ix * self.window_size - self.window_size / 2))
            end_frame = min(signal_crop.shape[0], ix * self.window_size + self.window_size / 2)

            start_s, end_s = start_frame / sr_final, end_frame / sr_final

            # Get the labels
            df_notes_in_window = self._get_rows_in_range(df_annots=df_notes_in_crop, start_s=start_s, end_s=end_s)
            onsets_in_window, notes_in_window = df_notes_in_window.OnsetTime, df_notes_in_window.MidiPitch
            labels.append(len(onsets_in_window) > 0)  # If there is at least one onset in the window, mark 1, else 0.
            note_vectors.append(self._get_multi_hot_notes(notes_in_window)[:, np.newaxis])

        labels = np.array(labels).astype(np.long)[np.newaxis, ...]
        note_vectors = np.hstack(note_vectors).astype(np.float32)  # each element represents the probability of the particular note in the resp window
        assert labels.shape[1] == n_windows and labels.shape[0] == 1
        assert np.all(note_vectors.shape == np.array([N_NOTES, n_windows]))
        assert np.all(note_vectors.any(axis=0) == labels[:])  # Only have notes marked in windows with onsets

        return {
            # signal related info
            'signal': signal_crop,
            'sr': sr_final,
            'start_s': start_s_crop,
            'end_s': end_s_crop,

            'onsets': target_onsets,

            # model inputs and targets
            'features': features[np.newaxis, ...],  # 1 * n_windows * n_feats
            'labels': labels,  # 1 * n_windows
            'notes': note_vectors,  # N_NOTES * n_windows

            # for debugging
            'file_path': str(annot_file_path),
        }

    @staticmethod
    def _get_multi_hot_notes(notes: List[int]) -> np.ndarray:
        notes = np.unique(notes)
        assert np.all(notes <= 108) and np.all(notes >= 21)
        notes -= 21

        multi_hot = np.zeros(N_NOTES)
        multi_hot[notes] = 1
        return multi_hot

    @staticmethod
    def _get_rows_in_range(df_annots: pd.DataFrame, start_s: float, end_s: float, col: str = 'OnsetTime',
                           relative: bool = False) -> pd.DataFrame:
        inds = (start_s <= df_annots[col]) & (df_annots[col] <= end_s)
        df_sub = df_annots.loc[inds].copy()
        if relative:
            df_sub.loc[:, ['OnsetTime', 'OffsetTime']] -= start_s
        return df_sub

    @staticmethod
    def _get_random_signal_crop(signal, sr, crop_len_sec) -> Tuple[np.ndarray, float, float]:
        n_total_frames = signal.shape[0]
        n_frames_in_crop = crop_len_sec * sr

        start_frame = np.random.randint(0, max(n_total_frames - n_frames_in_crop, 1))
        end_frame = min(start_frame + n_frames_in_crop, n_total_frames)

        signal_sample = signal[start_frame: end_frame]
        start_s, end_s = start_frame / sr, end_frame / sr

        return signal_sample, start_s, end_s

    @staticmethod
    def _load_annotations_df(annot_file_path: str) -> pd.DataFrame:
        if not os.path.exists(annot_file_path):
            raise FileNotFoundError(str(annot_file_path))

        df_annot = pd.read_csv(annot_file_path, sep='\t', )
        df_annot.loc[:, 'OnsetTime'] = np.round(df_annot.OnsetTime, decimals=2)

        return df_annot

    def __len__(self):
        return len(self.file_list)


if __name__ == '__main__':
    dataset = SignalWindowDataset(
        folder_path='../data',
    )

    np.random.seed(10)
    datum = dataset[0]
    print(datum['labels'])
