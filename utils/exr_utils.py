import numpy as np
import pims
import cv2
import random
import shutil
import re
from subprocess import Popen, PIPE, STDOUT, run
import tqdm
import os
from PIL import Image
from importlib import reload
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
reload(cv2)


def get_circular_mask_4_img(img: np.ndarray, scale_radius: float = 1.0) -> np.ndarray:
    """
    Returns a mask of the same size as img with a circular mask of 1 where the endoscopic image is.
    :param img: endoscopic image
    :param scale_radius: optional resize of the found circular mask.
    :raise ImageCroppingException:  when a proper circle can't be found.
    :return: a boolean array with True values within the circular mask
    """
    gray_img = rgb2gray(img)
    ret, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    img_height = thresh_img.shape[0]
    img_width = thresh_img.shape[1]
    points = []
    # go through lines with small steps to find points that may be the contour of the circle
    for y in range(0, img_height, 5):
        # get index pairs for consecutive runs of True values
        idx_pairs = np.where(np.diff(np.hstack(([False], thresh_img[y] == 255, [False]))))[0].reshape(-1, 2)
        # assert that runs have been found
        if len(idx_pairs) == 0:
            continue
        run_lengths = np.diff(idx_pairs, axis=1)
        # assert that there is only one "long" run
        if len(idx_pairs) > 1:
            if np.sort(run_lengths)[1] > 20:
                continue
        x1, x2 = idx_pairs[run_lengths.argmax()]  # Longest island
        run_length = x2 - x1
        # filter out short runs like for text etc.
        if run_length < 0.2 * img_width:
            continue
        points = points + [(x1, y), (x2, y)]
    n_samples = 15
    if len(points) < n_samples * 3:
        raise ImageCroppingException(img, "Not enough samples to process frame")
    max_circle = get_biggest_circle(points, n_samples)
    return create_circular_mask(h=img_height, w=img_width, center=max_circle[0:2], radius=max_circle[-1] * scale_radius)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)


def rgb2luminance(rgb):
    return np.dot(rgb[..., :3], [0.2126, 0.7152, 0.0722]).astype(np.uint8)


def draw_circles(img, circles):
    res = np.copy(img)
    for (x, y, r) in circles[:5]:
        cv2.circle(res, (x, y), r, (0, 255, 0), 4)
    return res


def crop_img_opencv(img, size=256, real_depth=None):
    height = img.shape[0]
    max_height = 512
    # only shrink if img is bigger than required
    if max_height < height:
        # get scaling factor
        scaling_factor = max_height / float(height)
        # resize image
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        real_depth = cv2.resize(real_depth, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    gray_img = rgb2gray(img)
    black_val = np.mean(gray_img[:20, :20])
    # threshold = black_val+2
    ret, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    filter_error = filter(img[thresh_img == 255])
    # if filter_error is not None:
    #     raise ImageCroppingException(img, filter_error)
    img_height = thresh_img.shape[0]
    img_width = thresh_img.shape[1]
    points = []
    # go through lines with small steps to find points that may be the contour of the circle
    for y in range(0, img_height, 5):
        # get index pairs for consecutive runs of True values
        idx_pairs = np.where(np.diff(np.hstack(([False], thresh_img[y] == 255, [False]))))[0].reshape(-1, 2)
        # assert that runs have been found
        if len(idx_pairs) == 0:
            continue
        run_lengths = np.diff(idx_pairs, axis=1)
        # assert that there is only one "long" run
        if len(idx_pairs) > 1:
            if np.sort(run_lengths)[1] > 20:
                continue
        x1, x2 = idx_pairs[run_lengths.argmax()]  # Longest island
        # x2 = img_width - np.argmax(np.flip(thresh_img[y])) - 1 
        run_length = x2 - x1
        # filter out short runs like for text etc.
        if run_length < 0.2 * img_width:
            continue
        points = points + [(x1, y), (x2, y)]
    n_samples = 15
    if len(points) < n_samples * 3:
        raise ImageCroppingException(img, "Not enough samples to process frame")
    max_circle = get_biggest_circle(points, n_samples)
    img_copy = np.copy(img)
    # cv2.circle(img_copy,max_circle[0:2],max_circle[2],(0,255,0),thickness = 10)
    x_low = max(0, max_circle[0] - max_circle[2])
    y_low = max(0, max_circle[1] - max_circle[2])
    x_high = min(img_width - 1, max_circle[0] + max_circle[2])
    y_high = min(img_height - 1, max_circle[1] + max_circle[2])
    img_copy = img_copy[y_low:y_high, x_low:x_high, :]
    real_depth_copy = real_depth.copy()[y_low:y_high, x_low:x_high, :]
    img_square = squarify(img_copy, black_val)
    real_depth_square = squarify(real_depth_copy, 0)
    # circle_mask = create_circular_mask(h=img_height, w=img_width, center=max_circle[0:2], radius=max_circle[-1])
    # circle_mask = squarify(circle_mask[y_low:y_high, x_low:x_high], 0).astype(np.int8)
    # if not blur_check(img_square, circle_mask):
    #     raise ImageCroppingException(img, "Image too blurry")
    return cv2.resize(img_square, (size, size)), cv2.resize(real_depth_square, (size, size))


def create_circular_mask(h, w, center=None, radius=None) -> np.ndarray:
    """ https://stackoverflow.com/a/44874588 """
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def blur_check(image, mask, threshold: float = 50) -> bool:
    """
    Check if an image is too blurry
    :param image: the image
    :param mask: a mask to consider which pixels
    :return: True if the image is considered "not blurry"
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = laplacian[np.where(mask)].var()
    return laplacian > threshold


def squarify(M, pad_constant):
    (a, b) = M.shape[0:2]
    if a > b:
        pad_val = a - b
        pad_left = pad_val // 2
        pad_right = pad_val - pad_left
        padding = ((0, 0), (pad_left, pad_right), (0, 0))
    else:
        pad_val = b - a
        pad_top = pad_val // 2
        pad_bot = pad_val - pad_top
        padding = ((pad_top, pad_bot), (0, 0), (0, 0))

    if len(M.shape) < 3:
        padding = padding[0:2]

    return np.pad(M, tuple(padding), mode='constant', constant_values=pad_constant)


def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return None

    # Center of circle
    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0]) ** 2 + (cy - p1[1]) ** 2)
    return [cx, cy, radius]


def reject_outliers(data, m=2):
    d = np.abs(data - np.median(data, axis=0))
    mdev = np.median(d, axis=0)
    s = np.divide(d, mdev, out=np.zeros_like(data), where=mdev != 0)
    return np.all(s < m, axis=1)


def exr_2_numpy(exr_file: str) -> np.ndarray:
    """
    read an exr file using cv2.
    Note: if the file is depth from blender, it could be the same value repeated 3 times (rgb).

    :param exr_file: path to the file
    :return: the cv2 imported image
    """
    return cv2.imread(exr_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

# def exr2numpy(exr, factor=1000, maxvalue=1., normalize=True):
#     """ converts 1-channel exr-data to 2D numpy arrays """
#     file = OpenEXR.InputFile(exr)
#
#     # Compute the size
#     dw = file.header()['dataWindow']
#     sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
#
#     # Read the three color channels as 32-bit floats
#     FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
#     (R) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in "R"]
#
#     # create numpy 2D-array
#     img = np.zeros((sz[1], sz[0], 3), np.float32)
#
#     # normalize
#     data = np.array(R)
#     data[data > maxvalue] = maxvalue
#
#     if normalize:
#         data /= np.max(data)
#
#     img = np.array(data).reshape(img.shape[0], -1)
#
#     return img * factor
#
#

def get_exr_max_depth(filepath):
    path = filepath + ".exr"
    img = exr_2_numpy(path)
    return np.max(img)


def get_biggest_circle(points, n_samples=20):
    n_points = 3 * n_samples
    samples = random.sample(points, n_points)
    samples = np.reshape(samples, newshape=(n_samples, 3, 2))
    circles = []
    for point_set in samples:
        circle = define_circle(*point_set)
        if circle is not None:
            circles.append(circle)
    circles = np.array(circles)
    circles = circles[reject_outliers(circles)]
    max_circle_idx = np.argmax(circles[:, 2])
    max_circle = circles[max_circle_idx, :].astype(int)
    return max_circle


def deinterlace_check(video_path: str) -> bool:
    """
    returns True if video needs to be de-interlaced
    :param video_path:
    """
    check = shutil.which('ffmpeg')
    if not check:
        raise ValueError("ffmpeg not found in PATH")
    output = '/dev/null' if not os.name == 'nt' else 'NUL'
    p = run(
        ["ffmpeg", "-filter:v", 'idet', '-frames:v', '360', '-an', '-f', 'rawvideo', '-y', output, '-i', video_path],
        capture_output=True)
    regex_str = re.compile('.*Single.*TFF:.* (\d+).*BFF.*Progressive:.* (\d+).*Undetermined.*')
    # IMPORTANT: MAY NEED TO READ FROM STDOUT IF ON WINDOWS
    frame_counts = re.findall(regex_str, p.stderr.decode('utf-8'))
    if not frame_counts:
        return False
    tff, prog = int(frame_counts[0][0]), int(frame_counts[0][1])
    if tff == 0:
        percent = 100
    else:
        percent = 100 * (prog / (prog + tff))
    deinterlacing_required = percent < 99

    return deinterlacing_required


def get_video_length_ffprobe(video_file: str) -> int:
    """
    Uses ffprobe to get the length of a video
    :param video_file: path to file
    :return: length of video in milliseconds
    """
    extract_reg = re.compile('.*\nduration=(\d+\.\d+).*')
    p = run(["ffprobe", "-i", video_file, '-show_entries', 'format=duration', '-v', 'quiet'], capture_output=True)
    result = p.stdout.decode('utf-8')
    duration = float(re.findall(extract_reg, result)[0])
    return int(duration * 1000)


def deinterlace(video_file: str) -> str:
    if not deinterlace_check(video_file):
        return video_file
    filename = os.path.basename(video_file)
    new_name = f'{os.path.splitext(filename)[0]}_original.mp4'
    new_filepath = os.path.join(os.path.dirname(video_file), new_name)
    os.rename(video_file, new_filepath)
    # do a bunch of converting names from mpg to mp4 and back because it seemed to affect ffmpeg results
    p = Popen(['/usr/bin/ffmpeg', '-i', new_filepath, '-vf', 'yadif=1:parity=tff', video_file[:-3] + "mp4"],
              stdout=PIPE,
              stderr=STDOUT)
    output = iter(lambda: p.stdout.read(1).decode('ascii'), str)
    buffer = ''
    loadingbar = tqdm.tqdm(total=get_video_length_ffprobe(new_filepath))
    time_regex = re.compile('.*time=(\d+):(\d+):(\d+).(\d+).*')
    last_ms = 0
    while p.poll() is None:
        c = next(output)
        if c != '\r':
            buffer += c
        else:
            timestamp = re.findall(time_regex, buffer)
            if timestamp:
                timestamp = timestamp[0]
                h = float(timestamp[0])
                m = float(timestamp[1])
                s = float(timestamp[2]) + float(timestamp[3]) / 100
                ms = int(h * 60 * 60 * 1000 + m * 60 * 1000 + s * 1000)
                loadingbar.update(ms - last_ms)
                last_ms = ms
            buffer = ''
    os.rename(video_file[:-3] + "mp4", video_file)
    loadingbar.close()
    return video_file


def extract_frames(video, target_dir, scenes, size, failed_frames_target_dir):
    if scenes is None:
        return
    full_video_path = video
    full_video_path = deinterlace(full_video_path)
    video_name = os.path.basename(full_video_path)
    vid = pims.PyAVReaderIndexed(full_video_path)
    n_frames = len(vid)
    step_width = 5
    if len(scenes) == 0:
        scenes.append((0, len(vid) - 1))
    for scene in scenes:
        if not isinstance(scene, int):
            start, end = scene
            indices = np.arange(start, min(n_frames, end + 1), step_width)
        else:
            indices = [scene]
        for frame_idx in indices:
            if (frame_idx / step_width) % 100 == 0:
                print("Processing Frame {}".format(frame_idx))
            try:
                img = vid[frame_idx]
                # cropped_img = crop_img(img)
                cropped_img = crop_img_opencv(img, size)
                # if np.mean(rgb2gray(cropped_img)) > 40:
                save_frame(cropped_img, target_dir, video_name, frame_idx)
            except ImageCroppingException as err:
                print("Image with idx {} cropping failed, saving it and skipping to next frame. Error: {}".format(
                    frame_idx, err))
                save_frame(err.img, failed_frames_target_dir, video_name, frame_idx)
            except ValueError as err:
                print("Image with idx {} could not find any video packets,skipping to next frame. Error: {}".format(
                    frame_idx, err))
    vid.close()


def filter(img_vals: np.ndarray):
    luminance_vals = rgb2luminance(img_vals)
    if np.sum(luminance_vals > 20) < 0.5 * len(luminance_vals):
        return "too dark"
    # pil_img = Image.fromarray(img)
    # dom_colors = np.array(sorted(pil_img.getcolors(pil_img.size[0]*pil_img.size[1]),reverse=True), np.dtype('int'))[:12]
    # dom_colors = np.array(dom_colors, np.dtype('int'))[:,1]

    num_colors = 100
    dom_colors, counts = np.unique(img_vals, axis=0, return_counts=True)
    dom_colors = dom_colors[np.argsort(counts)[::-1]][:num_colors]
    if np.sum(dom_colors[:, 0] > np.mean(dom_colors[:, 1:3], axis=1)) < 0.5 * num_colors:
        return "not enough red"
    return None


def save_frame(frame, video_path, video_name, frame_idx, size=None):
    pil_img = Image.fromarray(frame)
    if size is not None:
        pil_img = pil_img.resize((size, size))
    filename = "{}-{}.png".format(os.path.splitext(video_name)[0], str(frame_idx))
    pil_img.save(os.path.join(video_path, filename))


class ImageCroppingException(Exception):
    """Raised when an edoscop image can not be cropped"""

    def __init__(self, img, message="Image cropping failed"):
        og_height = img.shape[0]
        og_width = img.shape[1]
        width = 200
        height = int(og_height / og_width * width)
        self.img = cv2.resize(img, (width, height), cv2.INTER_AREA)
        self.message = message
        super().__init__(self.message)
