import os
from pathlib import Path
from typing import Callable, Optional, List
from tqdm import tqdm
import numpy as np
import imageio.v3 as iio
import trimesh
import pyrender

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

def render_mesh_on_video(video, mesh, poses, K, use_texture=True, verbose=True):
    """
    Renders a 3D mesh onto a single image or a video sequence.
    
    Args:
        video: ndarray of shape (H, W, 3) or (T, H, W, 3).
        mesh: trimesh.base.Trimesh object.
        poses: ndarray of shape (4, 4) or (T, 4, 4) representing 6D poses.
        K: ndarray of shape (3, 3) representing camera intrinsics.
        verbose: bool, whether to display the tqdm progress bar.
        use_texture: bool, whether to use the mesh's original texture.
    
    Returns:
        rendered_image: ndarray with the same shape as the input image.
    """
    
    # 1. Dimension alignment: Normalize input to (T, H, W, 3) and (T, 4, 4)
    is_video = video.ndim == 4
    if not is_video:
        video = video[np.newaxis, ...]
        poses = poses[np.newaxis, ...]
    
    T, H, W, _ = video.shape
    
    # 2. Initialize pyrender scene
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])
    
    # Material handling
    if use_texture and hasattr(mesh.visual, 'material'):
        mesh = pyrender.Mesh.from_trimesh(mesh)
    else:
        # Fallback to a gray metallic material
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.5, 0.5, 0.5, 1.0],
            metallicFactor=0.2,
            roughnessFactor=0.8
        )
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        
    mesh_node = scene.add(mesh)
    
    # Configure camera intrinsics
    # pyrender uses fx, fy, cx, cy for its IntrinsicsCamera
    camera = pyrender.IntrinsicsCamera(
        fx=K[0, 0], fy=K[1, 1],
        cx=K[0, 2], cy=K[1, 2]
    )
    
    # Transformation matrix: OpenCV (x-right, y-down, z-forward) 
    # to OpenGL (x-right, y-up, z-backward)
    cv_to_gl = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])
    
    # Add camera at the origin
    scene.add(camera, pose=np.eye(4))
    
    # Add directional lighting
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=np.eye(4))

    # 3. Initialize the offscreen renderer
    renderer = pyrender.OffscreenRenderer(W, H)
    
    output_frames = []
    
    # 4. Main rendering loop
    pbar = tqdm(range(T), disable=not verbose, desc="Rendering Mesh")
    for i in pbar:
        # Update object pose with coordinate system transformation
        # Assuming input poses are Object-to-Camera in OpenCV convention
        current_pose = cv_to_gl @ poses[i]
        scene.set_pose(mesh_node, pose=current_pose)
        
        # Perform rendering
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        
        # 5. Image composition (Overlaying rendered mesh on original image)
        # Create a mask where the object is present (depth > 0)
        mask = depth > 0
        img_frame = video[i].copy()
        
        # Apply the rendered pixels to the original frame
        img_frame[mask] = color[mask, :3]
        output_frames.append(img_frame)
        
    renderer.delete()
    
    # 6. Restore original dimensions
    result = np.stack(output_frames, axis=0)
    return result if is_video else result[0]