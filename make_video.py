import time
from argparse import ArgumentParser
from random import random

import cv2
import numpy as np

from src.DeepDream import DeepDream

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        help="Video file to make the deep dream from.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        help="Path to save generated video.",
        required=True,
    )
    parser.add_argument(
        "--start_second",
        dest="start_second",
        type=float,
        default=0,
        required=False,
    )
    parser.add_argument(
        "--end_second",
        dest="end_second",
        type=float,
        default=None,
        required=False,
    )

    args = parser.parse_args()
    input_video_path = args.input_file
    output_video_path = args.output_file
    start_second = args.start_second
    end_second = args.end_second

    cap = cv2.VideoCapture(input_video_path)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, FPS, (frame_width, frame_height))

    dreamer = DeepDream()

    start = time.time()
    current_frame = 0
    first_frame = int(FPS * start_second)
    last_frame = int(FPS * end_second) if end_second else None

    prev_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame += 1
        if current_frame < first_frame:
            continue

        frame_rgb = frame[:, :, ::-1]
        frame_array = np.array(frame_rgb, dtype=np.float16)
        if prev_frame is not None:
            blend = random()
            frame_array = (1 - blend) * frame_array + blend * prev_frame
        processed_frame = dreamer.dream(frame_array, step_size=0.01, steps_per_octave=10, n_octaves=5)
        prev_frame = processed_frame

        out.write(processed_frame[:, :, ::-1])

        if (current_frame - first_frame) % round(FPS) == 0:
            print(f"Processed {round((current_frame - first_frame) / FPS)} seconds")
            print(f"Elapsed time = {round((time.time() - start) / 60)} min")

        if last_frame and current_frame == last_frame:
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
