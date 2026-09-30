import os
import io
import glob
import shutil
import subprocess
import uuid
import base64
import cv2
import glob
import traceback
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from PIL import Image
from schemas.input import INPUT_SCHEMA

# Use /workspace for the Docker volume mount path
# otherwise use the current working directory
worker_build_env = os.getenv('WORKER_BUILD_ENV')
if worker_build_env:
    VOLUME_PATH = '/workspace'
else:
    VOLUME_PATH = os.getcwd()

GPU_ID = 0
TMP_PATH = f'{VOLUME_PATH}/tmp'
MODELS_PATH = f'{VOLUME_PATH}/models/ESRGAN'
GFPGAN_MODEL_PATH = f'{VOLUME_PATH}/models/GFPGAN/GFPGANv1.3.pth'
logger = RunPodLogger()


# ---------------------------------------------------------------------------- #
# Application Functions                                                        #
# ---------------------------------------------------------------------------- #
def upscale(
        source_image_path,
        image_extension,
        model_name='RealESRGAN_x4plus',
        outscale=4,
        face_enhance=False,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        denoise_strength=0.5
):
    """
    model_name options:
        - RealESRGAN_x4plus
        - RealESRNet_x4plus
        - RealESRGAN_x4plus_anime_6B
        - RealESRGAN_x2plus
        - realesr-animevideov3
        - realesr-general-x4v3

    image_extension: .jpg or .png

    outscale: The final upsampling scale of the image

    face_enhance: Whether or not to enhance the face

    tile: Tile size, 0 for no tile during testing

    tile_pad: Tile padding (default = 10)

    pre_pad: Pre padding size at each border

    denoise_strength: 0 for weak denoise (keep noise)
                      1 for strong denoise ability
                      Only used for the realesr-general-x4v3 model
    """

    # determine models according to model names
    model_name = model_name.split('.')[0]

    if image_extension == '.jpg':
        image_format = 'JPEG'
    elif image_extension == '.png':
        image_format = 'PNG'
    else:
        raise ValueError(f'Unsupported image type, must be either JPEG or PNG')

    if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
    # TODO: Implement these
    # elif model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
    #     model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    #     netscale = 4
    #     file_url = [
    #         'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
    #         'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
    #     ]
    elif model_name == '4x-UltraSharp':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'lollypop':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    # determine model paths
    model_path = os.path.join(MODELS_PATH, model_name + '.pth')

    if not os.path.isfile(model_path):
        raise Exception(f'Could not find model: {model_path}')

    # use dni to control the denoise strength
    dni_weight = None
    # if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
    #     wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
    #     model_path = [model_path, wdn_model_path]
    #     dni_weight = [denoise_strength, 1 - denoise_strength]

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half,
        gpu_id=GPU_ID
    )

    if face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path=GFPGAN_MODEL_PATH,
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )

    img = cv2.imread(source_image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise RuntimeError(f'Source image ({source_image_path}) is corrupt')

    try:
        if face_enhance:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as e:
        raise RuntimeError(e)
    else:
        result_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format=image_format)
        image_data = output_buffer.getvalue()
        return base64.b64encode(image_data).decode('utf-8')


def is_video(data):
    """MP4 carries 'ftyp' at byte 4; WebM opens with the EBML magic."""
    return data[4:8] == b'ftyp' or data[:4] == b'\x1a\x45\xdf\xa3'


def probe_frame_rate(video_path):
    """Frames are reassembled at the source rate, so it has to come off the source."""
    rate = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', video_path],
        capture_output=True, text=True, check=True
    ).stdout.strip()
    num, _, den = rate.partition('/')
    return float(num) / float(den or 1)


def upscale_video(
        source_video_path,
        work_dir,
        model_name,
        outscale,
        face_enhance,
        tile,
        tile_pad,
        pre_pad,
        half
):
    """
    Upscales every frame of a clip and writes it back as video.

    Sending a whole clip in one request is what makes this worth having: per frame, the
    round trip costs more than the inference does, so 57 separate calls spend most of
    their time moving base64 rather than upscaling.
    """
    frames_in = os.path.join(work_dir, 'in')
    frames_out = os.path.join(work_dir, 'out')
    os.makedirs(frames_in, exist_ok=True)
    os.makedirs(frames_out, exist_ok=True)

    fps = probe_frame_rate(source_video_path)
    subprocess.run(
        ['ffmpeg', '-v', 'error', '-i', source_video_path,
         os.path.join(frames_in, '%06d.png')],
        check=True
    )

    frames = sorted(glob.glob(os.path.join(frames_in, '*.png')))
    if not frames:
        raise RuntimeError('No frames could be extracted from the video')

    logger.info(f'Upscaling {len(frames)} frames at {fps:.2f} fps')

    for index, frame_path in enumerate(frames, start=1):
        encoded = upscale(
            frame_path, '.png', model_name, outscale,
            face_enhance, tile, tile_pad, pre_pad, half
        )
        with open(os.path.join(frames_out, os.path.basename(frame_path)), 'wb') as out_file:
            out_file.write(base64.b64decode(encoded))

        if index % 10 == 0 or index == len(frames):
            logger.info(f'Upscaled {index}/{len(frames)} frames')

    output_path = os.path.join(work_dir, 'output.mp4')
    subprocess.run(
        ['ffmpeg', '-v', 'error', '-y', '-framerate', str(fps),
         '-i', os.path.join(frames_out, '%06d.png'),
         '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', output_path],
        check=True
    )

    with open(output_path, 'rb') as video_file:
        return base64.b64encode(video_file.read()).decode('utf-8'), len(frames), fps


def determine_file_extension(image_data):
    image_extension = None

    try:
        if image_data.startswith('/9j/'):
            image_extension = '.jpg'
        elif image_data.startswith('iVBORw0Kg'):
            image_extension = '.png'
        else:
            # Default to png if we can't figure out the extension
            image_extension = '.png'
    except Exception as e:
        image_extension = '.png'

    return image_extension


def upscaling_api(input):
    if not os.path.exists(TMP_PATH):
        os.makedirs(TMP_PATH)

    unique_id = uuid.uuid4()
    source_image_data = input['source_image']
    model_name = input['model']
    outscale = input['scale']
    face_enhance = input['face_enhance']
    tile = input['tile']
    tile_pad = input['tile_pad']
    pre_pad = input['pre_pad']
    half = input['half']

    # Decode the source image data
    source_image = base64.b64decode(source_image_data)

    # A clip takes the video path: extract, upscale every frame, reassemble.
    if is_video(source_image):
        work_dir = f'{TMP_PATH}/video_{unique_id}'
        os.makedirs(work_dir, exist_ok=True)
        source_video_path = os.path.join(work_dir, 'source.mp4')

        with open(source_video_path, 'wb') as source_file:
            source_file.write(source_image)

        try:
            encoded, frame_count, fps = upscale_video(
                source_video_path, work_dir, model_name, outscale,
                face_enhance, tile, tile_pad, pre_pad, half
            )
        except Exception as e:
            logger.error(f'An exception was raised: {e}')
            shutil.rmtree(work_dir, ignore_errors=True)

            return {
                'error': traceback.format_exc(),
                'refresh_worker': True
            }

        # The frames alone can run to hundreds of megabytes; the worker outlives the job.
        shutil.rmtree(work_dir, ignore_errors=True)

        return {
            'status': 'ok',
            'video': encoded,
            'frames': frame_count,
            'fps': fps
        }

    source_file_extension = determine_file_extension(source_image_data)
    source_image_path = f'{TMP_PATH}/source_{unique_id}{source_file_extension}'

    # Save the source image to disk
    with open(source_image_path, 'wb') as source_file:
        source_file.write(source_image)

    try:
        result_image = upscale(
            source_image_path,
            source_file_extension,
            model_name,
            outscale,
            face_enhance,
            tile,
            tile_pad,
            pre_pad,
            half
        )
    except Exception as e:
        logger.error(f'An exception was raised: {e}')

        return {
            'error': traceback.format_exc(),
            'refresh_worker': True
        }

    # Clean up temporary images
    os.remove(source_image_path)

    return {
        'image': result_image
    }


# ---------------------------------------------------------------------------- #
# RunPod Handler                                                               #
# ---------------------------------------------------------------------------- #
def handler(event):
    validated_input = validate(event['input'], INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {
            'errors': validated_input['errors']
        }

    return upscaling_api(validated_input['validated_input'])


if __name__ == "__main__":
    logger.info('Starting RunPod Serverless...')
    runpod.serverless.start(
        {
            'handler': handler
        }
    )
