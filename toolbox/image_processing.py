import os
from pathlib import Path
from typing import Callable, Optional, List
import numpy as np
import imageio.v3 as iio
from tqdm import tqdm

def images_to_video(
    img_dir: str,
    output_path: Optional[str] = None,
    fps: int = 30,
    sort_func: Optional[Callable] = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Synthesizes a sequence of images from a directory into a video file.

    Args:
        img_dir: Path to the directory containing images.
        output_path: Path to save the output video. If None, the video is not saved.
        fps: Frames per second. Defaults to 30.
        sort_func: Custom sorting function for file list. Defaults to alphabetical.
        verbose: If True, prints status info,

    Returns:
        np.ndarray: A 4D numpy array representing the video (T, H, W, C).
    """
    img_path = Path(img_dir)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # 1. search images
    files = [
        str(f) for f in img_path.iterdir() 
        if f.suffix.lower() in extensions
    ]
    
    if not files:
        raise FileNotFoundError(f"No images found in {img_dir}.")

    if verbose:
        print(f"[INFO] Found {len(files)} images.")

    # 2. sort files
    if sort_func:
        files = sort_func(files)
    else:
        files.sort()

    # 3. read images
    frames = []
    
    for f in tqdm(files, desc="Processing Frames", disable=not verbose):
        frame = iio.imread(f)
        frames.append(frame)
    
    video_array = np.stack(frames, axis=0)

    # 4. save video
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        iio.imwrite(
            output_path, 
            video_array, 
            fps=fps,
        )
        
        if verbose:
            print(f"[INFO] Video successfully saved to: {os.path.abspath(output_path)}")
        
    return video_array