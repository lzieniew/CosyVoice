import sys
import os
import glob
import json
import subprocess
import tempfile

sys.path.insert(0, '/workspace/CosyVoice')
sys.path.insert(0, '/workspace/CosyVoice/third_party/Matcha-TTS')

import torch
import torchaudio


def download_base_model(base_model):
    """Download base model if not already cached. Returns local path."""
    if os.path.exists(base_model):
        return base_model
    from modelscope import snapshot_download
    print(f"Downloading base model: {base_model} ...")
    path = snapshot_download(base_model)
    print(f"Base model cached at: {path}")
    return path


def extract_weights(weights_path):
    """Extract checkpoint files from a directory or tar.gz. Returns path to directory with .pt files."""
    if not weights_path or not os.path.exists(weights_path):
        return None

    if os.path.isdir(weights_path):
        pt_files = glob.glob(os.path.join(weights_path, '*.pt'))
        if pt_files:
            return weights_path
        subdirs = glob.glob(os.path.join(weights_path, '*/*.pt'))
        if subdirs:
            return os.path.dirname(subdirs[0])
        return weights_path

    if weights_path.endswith(('.tar.gz', '.tgz')):
        tmp_dir = tempfile.mkdtemp(prefix='cosyvoice_weights_')
        subprocess.run(['tar', 'xzf', weights_path, '-C', tmp_dir], check=True)
        pt_files = glob.glob(os.path.join(tmp_dir, '**/*.pt'), recursive=True)
        if pt_files:
            return os.path.dirname(pt_files[0])
        return tmp_dir

    return None


def find_qwen_model(model):
    """Walk the model tree to find the Qwen2ForCausalLM instance and its parent.

    Returns (parent_object, attribute_name, qwen_model) so we can reassign if needed.
    """
    from transformers import Qwen2ForCausalLM

    # Walk all named modules to find the Qwen2ForCausalLM
    for name, module in model.named_modules():
        if isinstance(module, Qwen2ForCausalLM):
            print(f"  Found Qwen2ForCausalLM at: model.{name}")
            return name, module

    # Fallback: print the model tree for debugging
    print("  ERROR: Could not find Qwen2ForCausalLM. Model tree:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0 or len(name.split('.')) <= 3:
            print(f"    {name}: {type(module).__name__}")
    return None, None


def find_named_module(model, target_class_name):
    """Find a module by class name."""
    for name, module in model.named_modules():
        if type(module).__name__ == target_class_name:
            return name, module
    return None, None


def get_module_by_path(root, path):
    """Get a submodule by dot-separated path."""
    obj = root
    for attr in path.split('.'):
        if attr.isdigit():
            obj = obj[int(attr)]
        else:
            obj = getattr(obj, attr)
    return obj


def set_module_by_path(root, path, value):
    """Set a submodule by dot-separated path."""
    parts = path.split('.')
    parent = get_module_by_path(root, '.'.join(parts[:-1])) if len(parts) > 1 else root
    setattr(parent, parts[-1], value)


def apply_lora_manual(qwen_model, lora_state, alpha, r, device):
    """Merge LoRA weights directly into the base model parameters.

    No peft library needed — just pure math: W_new = W_orig + (B @ A) * (alpha / r)
    """
    scaling = alpha / r
    print(f"  LoRA scaling: alpha={alpha}, r={r}, scaling={scaling}")

    # Group LoRA A/B pairs by their base parameter path
    # Key format: base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
    # We need: model.layers.0.self_attn.q_proj (path inside Qwen2ForCausalLM)
    pairs = {}
    for key, tensor in lora_state.items():
        parts = key.split('.')
        # Find lora_A or lora_B in the key
        lora_idx = None
        for i, p in enumerate(parts):
            if p in ('lora_A', 'lora_B'):
                lora_idx = i
                break
        if lora_idx is None:
            continue

        # Strip the peft prefix (base_model.model.) to get the path relative to Qwen2ForCausalLM
        # Keys start with base_model.model. then the actual model path
        prefix_end = 0
        if parts[0] == 'base_model' and parts[1] == 'model':
            prefix_end = 2

        base_path = '.'.join(parts[prefix_end:lora_idx])
        lora_type = parts[lora_idx]  # 'lora_A' or 'lora_B'

        if base_path not in pairs:
            pairs[base_path] = {}
        pairs[base_path][lora_type] = tensor

    merged_count = 0
    for base_path, lora_pair in sorted(pairs.items()):
        if 'lora_A' not in lora_pair or 'lora_B' not in lora_pair:
            print(f"  WARNING: incomplete LoRA pair for {base_path}")
            continue

        try:
            module = get_module_by_path(qwen_model, base_path)
        except (AttributeError, IndexError) as e:
            print(f"  WARNING: could not find {base_path} in model: {e}")
            continue

        lora_A = lora_pair['lora_A'].to(device=device, dtype=module.weight.dtype)
        lora_B = lora_pair['lora_B'].to(device=device, dtype=module.weight.dtype)

        # W_new = W_orig + (B @ A) * scaling
        module.weight.data += (lora_B @ lora_A) * scaling
        merged_count += 1

    return merged_count


def apply_finetuned_weights(model, weights_dir):
    """Apply LoRA + fine-tuned decoder/embedding weights to a loaded CosyVoice model."""
    lora_path = os.path.join(weights_dir, 'lora_weights.pt')
    decoder_path = os.path.join(weights_dir, 'llm_decoder.pt')
    embedding_path = os.path.join(weights_dir, 'speech_embedding.pt')
    config_path = os.path.join(weights_dir, 'config.json')

    # Read LoRA config
    lora_r, lora_alpha = 32, 64
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f).get('config', {})
        lora_r = cfg.get('lora_r', lora_r)
        lora_alpha = cfg.get('lora_alpha', lora_alpha)

    # --- Apply LoRA weights ---
    if os.path.exists(lora_path):
        # Find the Qwen2ForCausalLM in the model tree
        qwen_path, qwen_model = find_qwen_model(model)
        if qwen_model is None:
            print("  FATAL: Cannot apply LoRA — Qwen2ForCausalLM not found in model!")
            return

        device = next(qwen_model.parameters()).device
        lora_state = torch.load(lora_path, map_location=device, weights_only=True)
        print(f"  Loaded lora_weights.pt: {len(lora_state)} tensors")

        merged = apply_lora_manual(qwen_model, lora_state, lora_alpha, lora_r, device)
        print(f"  Merged {merged} LoRA pairs into Qwen2 backbone")
    else:
        print(f"  No lora_weights.pt found")
        device = next(model.parameters()).device

    # --- Apply llm_decoder weights ---
    if os.path.exists(decoder_path):
        decoder_state = torch.load(decoder_path, map_location=device, weights_only=True)
        # Find llm_decoder by walking the model
        found = False
        for name, module in model.named_modules():
            if name.endswith('llm_decoder') and hasattr(module, 'weight'):
                module.load_state_dict(decoder_state)
                print(f"  Applied llm_decoder weights at: model.{name} (shape: {module.weight.shape})")
                found = True
                break
        if not found:
            print("  WARNING: llm_decoder not found in model!")

    # --- Apply speech_embedding weights ---
    if os.path.exists(embedding_path):
        embedding_state = torch.load(embedding_path, map_location=device, weights_only=True)
        found = False
        for name, module in model.named_modules():
            if name.endswith('speech_embedding') and hasattr(module, 'weight'):
                module.load_state_dict(embedding_state)
                print(f"  Applied speech_embedding weights at: model.{name} (shape: {module.weight.shape})")
                found = True
                break
        if not found:
            print("  WARNING: speech_embedding not found in model!")


def main():
    weights_path = os.environ.get('WEIGHTS', '')
    reference_wav = os.environ.get('REFERENCE', '')
    text_file = os.environ.get('TEXT', '')
    output_dir = os.environ.get('OUTPUT_DIR', '/output')
    base_model = os.environ.get('BASE_MODEL', 'FunAudioLLM/Fun-CosyVoice3-0.5B-2512')
    language = os.environ.get('LANGUAGE', 'You are a helpful assistant.<|endofprompt|>')
    prompt_text = os.environ.get('PROMPT_TEXT', '')
    speed = float(os.environ.get('SPEED', '1.0'))

    if not reference_wav or not os.path.isfile(reference_wav):
        sys.exit(f"ERROR: Reference wav not found: {reference_wav}")
    if not text_file or not os.path.isfile(text_file):
        sys.exit(f"ERROR: Text file not found: {text_file}")

    os.makedirs(output_dir, exist_ok=True)

    # Download base model
    model_dir = download_base_model(base_model)

    # Load base model
    from cosyvoice.cli.cosyvoice import AutoModel
    print(f"Loading model from: {model_dir}")
    model = AutoModel(model_dir=model_dir)

    # Apply fine-tuned LoRA weights if provided
    weights_dir = extract_weights(weights_path)
    if weights_dir:
        print(f"Applying fine-tuned weights from: {weights_dir}")
        apply_finetuned_weights(model.model.llm, weights_dir)

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

        # Prepend language/system prompt if not already present
        if line.startswith('<|') or line.startswith('You are'):
            text = line
        else:
            text = f"{language}{line}"

        # text_frontend=False: skip Chinese/English text normalization that mangles other languages
        results = model.inference_cross_lingual(
            text, reference_wav,
            stream=False, speed=speed, text_frontend=False)

        for result in results:
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
