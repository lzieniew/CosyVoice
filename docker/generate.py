import sys
import os
import glob
import shutil
import subprocess
import tempfile

sys.path.insert(0, '/workspace/CosyVoice')
sys.path.insert(0, '/workspace/CosyVoice/third_party/Matcha-TTS')

import torch
import torchaudio


def prepare_model_dir(base_model, weights_path):
    """Download base model and overlay fine-tuned weights.

    weights_path can be:
    - A directory containing .pt files
    - A single .pt file
    - A .tar.gz archive containing .pt files

    Returns path to a model directory ready for AutoModel.
    """
    from modelscope import snapshot_download

    if os.path.exists(base_model):
        base_path = base_model
    else:
        print(f"Downloading base model: {base_model} ...")
        base_path = snapshot_download(base_model)
        print(f"Base model cached at: {base_path}")

    if not weights_path:
        return base_path

    # Create working directory with symlinks to base model files
    work_dir = tempfile.mkdtemp(prefix='cosyvoice_model_')
    for item in os.listdir(base_path):
        src = os.path.join(base_path, item)
        dst = os.path.join(work_dir, item)
        os.symlink(src, dst)

    # Collect .pt files from the weights path
    pt_files = []

    if os.path.isdir(weights_path):
        # Directory: gather all .pt files inside it (recursively)
        pt_files = glob.glob(os.path.join(weights_path, '**/*.pt'), recursive=True)
    elif weights_path.endswith('.pt'):
        # Single .pt file
        pt_files = [weights_path]
    elif weights_path.endswith(('.tar.gz', '.tgz')):
        # Archive: extract and gather .pt files
        tmp_extract = tempfile.mkdtemp(prefix='cosyvoice_weights_')
        subprocess.run(['tar', 'xzf', weights_path, '-C', tmp_extract], check=True)
        pt_files = glob.glob(os.path.join(tmp_extract, '**/*.pt'), recursive=True)

    if not pt_files:
        print("WARNING: No .pt files found in the provided weights!")

    # Overlay .pt files onto the working model directory
    for pt_file in pt_files:
        basename = os.path.basename(pt_file)
        dst = os.path.join(work_dir, basename)
        if os.path.islink(dst) or os.path.exists(dst):
            os.unlink(dst)
        shutil.copy2(pt_file, dst)
        print(f"  Applied fine-tuned weights: {basename}")

    return work_dir


def main():
    weights_path = os.environ.get('WEIGHTS', '')
    reference_wav = os.environ.get('REFERENCE', '')
    text_file = os.environ.get('TEXT', '')
    output_dir = os.environ.get('OUTPUT_DIR', '/output')
    base_model = os.environ.get('BASE_MODEL', 'iic/Fun-CosyVoice3-0.5B')
    language = os.environ.get('LANGUAGE', '<|en|>')

    if not reference_wav or not os.path.isfile(reference_wav):
        sys.exit(f"ERROR: Reference wav not found: {reference_wav}")
    if not text_file or not os.path.isfile(text_file):
        sys.exit(f"ERROR: Text file not found: {text_file}")

    os.makedirs(output_dir, exist_ok=True)

    # Prepare model directory (download base + overlay fine-tuned weights)
    model_dir = prepare_model_dir(base_model, weights_path if weights_path and os.path.exists(weights_path) else None)

    from cosyvoice.cli.cosyvoice import AutoModel
    print(f"Loading model from: {model_dir}")
    model = AutoModel(model_dir=model_dir)

    # Read text segments
    with open(text_file, encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        sys.exit("ERROR: Text file is empty!")

    total = len(lines)
    all_speech = []
    segment_idx = 0

    for line_idx, line in enumerate(lines):
        print(f"\n[{line_idx + 1}/{total}] {line[:80]}{'...' if len(line) > 80 else ''}")

        # Prepend language tag if not already present
        text = line if line.startswith('<|') else f"{language}{line}"

        for result in model.inference_cross_lingual(text, reference_wav, stream=False):
            speech = result['tts_speech']
            seg_path = os.path.join(output_dir, f'segment_{segment_idx:04d}.wav')
            torchaudio.save(seg_path, speech, model.sample_rate)
            all_speech.append(speech)
            segment_idx += 1

    # Concatenate all segments into a single file
    if all_speech:
        combined = torch.cat(all_speech, dim=1)
        combined_path = os.path.join(output_dir, 'output.wav')
        torchaudio.save(combined_path, combined, model.sample_rate)
        duration = combined.shape[1] / model.sample_rate
        print(f"\nDone! {segment_idx} segments, {duration:.1f}s total -> {combined_path}")
    else:
        print("\nNo audio was generated.")


if __name__ == '__main__':
    main()
