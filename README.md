# Real-ESRGAN | RunPod Serverless Worker

The is the source code for a [RunPod](https://runpod.io?ref=2xxro4sy)
Serverless worker that uses [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
for Restoration/Upscaling.

> **This fork** adds two things to upstream: the `realesr-animevideov3` model, and video
> input. See [What this fork changes](#what-this-fork-changes).

## Model

The following models are available by default:

* RealESRGAN_x2plus
* RealESRGAN_x4plus
* RealESRNet_x4plus
* RealESRGAN_x4plus_anime_6B
* realesr-animevideov3 — compact per-frame network, built for video

## What this fork changes

**`realesr-animevideov3`.** Upstream's schema rejects the name even though its upscale
function supports it. Measured on ~58-frame clips it runs several times faster than the photo
networks and comes out cleaner on both animation and live action: it holds flat fills without
speckling them, where the photo networks invent texture that is not in the source.

**Video input.** `source_image` also takes a base64 MP4 or WebM. The worker extracts the
frames, upscales each one, and reassembles the clip at the source frame rate, answering with
`video` instead of `image`:

```json
{
  "status": "ok",
  "video": "<base64 mp4>",
  "frames": 57,
  "fps": 20.24
}
```

Sending a clip in one request is the point: per frame, the HTTP round trip costs more than
the inference. A 57-frame clip took 333 s as 57 separate calls against 33 s of actual work.

Audio is dropped — the frames are all that get upscaled. Mind the payload limits: RunPod caps
`/run` at 10 MB and `/runsync` at 20 MB, and base64 inflates by a third, so clips of more than
a few seconds need `s3Config` rather than an inline payload.

## Testing

1. [Local Testing](docs/testing/local.md)
2. [RunPod Testing](docs/testing/runpod.md)

## Building the Docker image that will be used by the Serverless Worker

1. [Building the Image](docs/building.md) 

## API

The worker provides an API for inference. The API payload looks like this:

```json
{
  "input": {
     "source_image": "base64 encoded source image content",
     "model": "RealESRGAN_x4plus",
     "scale": 2,
     "face_enhance": true
  }
}
```

## Serverless Handler

The serverless handler (`handler.py`) is a Python script that handles
inference requests.  It defines a function handler(event) that takes an
inference request, runs the inference using the Real-ESRGAN model, and
returns the output as a JSON response in the following format:

```json
{
  "output": {
    "status": "ok",
    "image": "base64 encoded output image"
  }
}
```

## Acknowledgements

- [Real-ESRGAN (ai-forever)](https://github.com/ai-forever/Real-ESRGAN)

## Community and Contributing

Pull requests and issues on [GitHub](https://github.com/ashleykleynhans/runpod-worker-real-esrgan)
are welcome. Bug fixes and new features are encouraged.

## Appreciate my work?

<a href="https://www.buymeacoffee.com/ashleyk" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>
