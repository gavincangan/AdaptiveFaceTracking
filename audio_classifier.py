import librosa
import pickle
import matplotlib as plt

sample_freq = 22050
audio_pickle_file = '/home/gavincangan/computerVision/AdaptiveFaceTracking/audio/audio_signal.pick'
audio_pickle_fp = open(audio_pickle_file, 'r+b')
audio_signal = pickle.load(audio_pickle_fp)
# print audio_signal.shape
video_frame_rate = 24
audio_index_max = 23304384

def frame_num_to_audio_index(frame_num):
    frame_time = float(frame_num) / video_frame_rate
    audio_index = round(int(frame_time * sample_freq))
    # print audio_index
    return audio_index


def get_audio_this_frame(this_frame_num):
    window_width = sample_freq / 1
    this_audio_index = frame_num_to_audio_index(this_frame_num)
    start_index = max(0, int(this_audio_index - window_width/2))
    start_index = min(start_index, audio_index_max - window_width - 1)
    end_index = start_index + window_width
    # print start_index, end_index, '  ',
    return audio_signal[start_index:end_index]

def read_audio():
    audio_filename = '/home/gavincangan/computerVision/AdaptiveFaceTracking/audio/TBBT_S10E16.mp3'
    audio_signal, sample_freq = librosa.load(audio_filename)
    # print sample_freq
    pickle.dump(audio_signal, audio_pickle_fp)

def frame_audio_fft(this_frame_num):
    this_frame_audio = get_audio_this_frame(this_frame_num)
    this_frame_fft = librosa.stft(this_frame_audio, n_fft=4096, win_length=4096)
    return this_frame_fft

if __name__ == '__main__':
    # read_audio()
    # audio_this_frame = get_audio_this_frame(103)
    fft_this_frame = frame_audio_fft(103)
    print fft_this_frame.shape
    # plt.plot()
