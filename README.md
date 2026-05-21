
## It's just a convenience fork of CosyVoice
This fork allows to easily run the model inside a docker container, with nvidia gpu passthrough, and reference voice for clonning.

Just follow those steps:
- 1. create `input/` and `output/` directories
```
mkdir input output
```
- 2. copy file with fine-tuned weights in it as `input/weights.tar.gz`
- 3. copy text you want to generate to `input/text.txt` - it can be quite long, just split it into chunks, each in it's own line
- 4. copy the reference voice file for cloning to `input/voice.wav`
- 5. run the container
```
docker compose up
```
It will be built on first run, subsequent runs of the container will use the already downloaded dependendencies and model
