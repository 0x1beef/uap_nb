import os

class FrameSequence:
    def __init__(self, frames_dir, fps):
        self.frames_dir = frames_dir
        self.fps = fps
        import glob
        self.num_frames = len(glob.glob(f'{self.frames_dir}/frame_*.png'))
        print(f"'{frames_dir}': {self.num_frames} frames @ {self.fps:.2f} fps")
    def get_frame(self, frame):
        import cv2
        return cv2.imread(f'{self.frames_dir}/frame_{frame:04d}.png', cv2.IMREAD_GRAYSCALE)
    def get_frame_count(self):
        return self.num_frames
    def get_frame_time(self, frame):
        return frame / self.fps
    def get_fps(self):
        return self.fps
    
def get_fps(video):
    command = f'ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate "{video}"'
    import subprocess, shlex, fractions
    fps_str = subprocess.check_output(shlex.split(command)).decode("utf-8").strip()
    return float(fractions.Fraction(fps_str))
    
def extract_frames(video, frames_dir, filters, verbose):
    if not os.path.exists(f'{frames_dir}/frame_0000.png'):
        # note: without the variable frame rate option this would produce some duplicate frames
        output_frames = f'-vsync vfr -start_number 0 {frames_dir}/frame_%04d.png'
        logs = '-hide_banner -loglevel error' if not verbose else ''
        cmd = f'ffmpeg {logs} -i "{video}" -vf "{filters}" {output_frames}'
        print(cmd)
        os.system(cmd)

def flir1_extract_frames(video, frames_dir, verbose):
    # note: crop can sometimes be off by a pixel, so make sure it's exact
    extract_frames(video, frames_dir, 'format=gray, crop=240:238:54:15:exact=1', verbose)

def navy_2015_extract_frames(video, frames_dir, verbose):
    # note: crop can sometimes be off by a pixel, so make sure it's exact
    extract_frames(video, frames_dir, 'format=gray, crop=428:428:104:27:exact=1', verbose)

def common_from_huggingface(dir, video_file, extract_frames_func, verbose):
    video = f'{dir}/{video_file}'
    if not os.path.exists(video):
        import utils
        utils.download_from_huggingface(f'logicbear/uap/{video}')
    extract_frames_func(video, dir, verbose)
    return FrameSequence(dir, get_fps(video))

def flir1_from_huggingface(verbose = False):
    return common_from_huggingface('flir1', '1 - FLIR.mp4', flir1_extract_frames, verbose)

def gimbal_from_huggingface(verbose = False):
    return common_from_huggingface('gimbal', '2 - GIMBAL.wmv', navy_2015_extract_frames, verbose)

def gofast_from_huggingface(verbose = False):
    return common_from_huggingface('gofast', '3 - GOFAST.wmv', navy_2015_extract_frames, verbose)

# interpolate values in a dataframe's fields around the switch from WH to BH
def gimbal_fix_wh_to_bh(df, fields, minus = 10, plus = 0):
    last_white_hot = 370
    first_black_hot = 372
    interp_range = range(last_white_hot - minus, first_black_hot + plus)
    import utils
    return utils.interpolate_rows(df, interp_range, fields)

def plot_frame_data(df, plot_func, x_range = None):
    import matplotlib.pyplot as plt
    if x_range is None:
        x_range = slice(df.first_valid_index(), df.last_valid_index())
    plot_func(df, x_range)
    ax = plt.gca()
    ax.set_xlabel('frames', loc='right')
    fps = df.attrs['fps']
    ax2 = ax.secondary_xaxis(location='top',
        functions=(lambda frame: frame / fps, lambda time: time * fps))
    ax2.set_xlabel('time [s]', loc='right')
    plt.grid()
    plt.legend()
    plt.show()

def get_frame_diff(frame, frame_diff_from, frame_diff_to, num_frames):
    return int(frame_diff_from + frame * (frame_diff_to + 1 - frame_diff_from) / num_frames)