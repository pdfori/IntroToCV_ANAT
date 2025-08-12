
import cv2
import matplotlib.pyplot as plt
import cv2


def get_frame_at_time(video_path, time_seconds):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame = int(time_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Could not read frame.")

def display_image(dts,title="",grey=0) :
    if grey:
        plt.imshow(dts, cmap="gray")
    else:
        plt.imshow(dts)
    plt.title(title)
    plt.axis("off")
    plt.show()



def show_color_channel_in_grey_scale(frame, channel):
    """
    Display one color channel of the frame as a grayscale image.

    :param frame: RGB image (H x W x 3)
    :param channel: 0 for Red, 1 for Green, 2 for Blue
    """
    color_dict = {0: "Red", 1: "Green", 2: "Blue"}
    gray_channel = frame[:, :, channel]
    plt.imshow(gray_channel, cmap='gray')
    plt.title(f"{color_dict[channel]} Channel (Grayscale)")

    plt.axis("off")
    plt.show()

def examine_row(frame, row_idx):
    frame_copy = frame.copy()
    frame_copy[row_idx, :, :] = [255, 0, 0]  # Mark row in red
    display_image(frame_copy, f"Row {row_idx} marked")

def examine_col(frame, col_idx):
    frame_copy = frame.copy()
    frame_copy[:,col_idx, :] = [255, 0, 0]  # Mark row in red
    frame_copy[:,(col_idx+1), :] = [255, 0, 0]  # Mark row in red
    frame_copy[:,(col_idx-1), :] = [255, 0, 0]  # Mark row in red
    display_image(frame_copy, f"COL {col_idx} marked")


def plot_row_grey_levels_by_channel(frame, color_ch, row_idx=0, title=None):
    color_dict = {0: "red",
                  1: "green",
                  2: "blue"}

    if not title:
        title = f"Gray Levels of Row {row_idx} ({color_dict[color_ch]} Channel)"

    # Plot gray levels of green channel
    green_channel = frame[:, :, color_ch]
    row_values = green_channel[row_idx, :]

    plt.figure(figsize=(10, 4))
    plt.plot(row_values, color=color_dict[color_ch], label=color_dict[color_ch])
    plt.title(title)
    plt.xlabel("Column Index")
    plt.ylabel(f"{color_dict[color_ch]} ch. Gray Level")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_col_grey_levels_by_channel(frame, color_ch, col_idx=0, title=None):
    color_dict = {0: "blue",
                  1: "green",
                  2: "red"}

    if not title:
        title = f"Gray Levels of Col {col_idx} ({color_dict[color_ch]} Channel)"

    # Plot gray levels of green channel
    green_channel = frame[:, :, color_ch]
    col_values = green_channel[:, col_idx]

    plt.figure(figsize=(10, 4))
    plt.plot(col_values, color=color_dict[color_ch], label=color_dict[color_ch])
    plt.title(title)
    plt.xlabel("Row Index")
    plt.ylabel(f"{color_dict[color_ch]} ch. Gray Level")
    plt.grid(True)
    plt.legend()
    plt.show()


def draw_roi(frame, row, col, dr, dc, color=(255, 0, 0), thickness=2):
    """
    Draws a rectangle around ROI defined by center (row, col) and half-sizes dr, dc.
    """
    top_left = (col - dc, row - dr)
    bottom_right = (col + dc, row + dr)
    frame_with_roi = frame.copy()
    cv2.rectangle(frame_with_roi, top_left, bottom_right, color, thickness)
    return frame_with_roi


def display_roi(frame,
                row,
                center,
                dr,
                dc,
                title="Re-Sized ROI"):
    roi = frame[row - dr: row + dr, center - dc: center + dc]
    display_image(roi, title)


def spatial_sample_columns(image, step, show=1):
    h, w, c = image.shape
    center = w // 2
    sampled_cols = [center]

    # Sample symmetrically around center
    for offset in range(1, w // step + 1):
        left = center - offset * step
        right = center + offset * step
        if 0 <= left:
            sampled_cols.append(left)
        if right < w:
            sampled_cols.append(right)

    sampled_cols = sorted(sampled_cols)
    sampled_image = image[:, sampled_cols, :]

    if show:
        # Mark sampled columns on original
        marked = image.copy()
        marked[:, sampled_cols, :] = [255, 0, 0]

        display_image(marked, "Original with Sampled Columns Marked")

    return sampled_image, sampled_cols


def spatial_sample_rows(image, step, show=1, thickness=3):
    h, w, c = image.shape
    center = h // 2
    sampled_rows = [center]

    # Sample symmetrically around center
    for offset in range(1, h // step + 1):
        top = center - offset * step
        bottom = center + offset * step
        if 0 <= top:
            sampled_rows.append(top)
        if bottom < h:
            sampled_rows.append(bottom)

    sampled_rows = sorted(sampled_rows)
    sampled_image = image[sampled_rows, :, :]

    if show:
        marked = image.copy()
        for r in sampled_rows:
            marked[max(0, r - thickness // 2):min(h, r + thickness // 2 + 1), :, :] = [255, 0, 0]

        display_image(marked, "Original with Sampled Rows Marked")

    return sampled_image, sampled_rows



def interpolate_to_original_width(sampled_img, original_width):
    # Resize using bilinear interpolation (default)
    h, _, c = sampled_img.shape
    restored = cv2.resize(sampled_img, (original_width, h), interpolation=cv2.INTER_LINEAR)
    return restored

def interpolate_to_original_height(sampled_img, original_height):
    # Resize using bilinear interpolation (default)
    _, w, c = sampled_img.shape
    restored = cv2.resize(sampled_img, (w, original_height), interpolation=cv2.INTER_LINEAR)
    return restored


import cv2


def video_to_frames(video_path, start_time, duration):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_time = start_time + duration
    end_frame = int(end_time * fps)

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def create_temporal_sampled_video(frames, output_path, fps=30, sample_interval=16):
    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 files
    h, w, c = frames[0].shape
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Sampling frames
    sampled_frames = frames[::sample_interval]  # Select frames at intervals of 16

    # Apply Zero-Order Hold Interpolation (repeat each frame 16 times)
    for frame in sampled_frames:
        for _ in range(sample_interval):
            out.write(frame)  # Repeat the frame

    out.release()




