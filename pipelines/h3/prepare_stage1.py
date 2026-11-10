"""Strip a preencoded dataset to H3 camera-only inputs and validate it.

Source JSONL: {"path":"cache.safetensors", "camera_path":"camera.safetensors"}.
The optional camera file supplies latent-aligned camera_viewmats and camera_K.
Qwen cache provenance must explicitly say qwen_skip_reference_videos=true.
"""
import argparse
import json
from pathlib import Path

from safetensors.torch import load_file, save_file

from h3.data import validate_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    keys = ("target_latents", "anchor_latents", "prompt_embeds", "text_token_tags",
            "camera_viewmats", "camera_K")
    with (output / "manifest.jsonl").open("w") as manifest:
        for index, line in enumerate(source.read_text().splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("qwen_skip_reference_videos") is not True:
                raise ValueError("Verify that cached Qwen embeds contain only first RGB + caption; "
                                 "reencode caches that include reference videos")
            tensors = load_file(str(source.parent / row["path"]))
            if row.get("camera_path"):
                tensors.update(load_file(str(source.parent / row["camera_path"])))
            sample = validate_sample({key: tensors[key] for key in keys})
            name = f"{index:08d}.safetensors"
            save_file({key: value.contiguous() for key, value in sample.items()}, str(output / name))
            manifest.write(json.dumps(dict(sample_id=row.get("sample_id", str(index)), path=name)) + "\n")
    (output / "SOURCE.json").write_text(json.dumps(dict(
        source_manifest=str(source), qwen_skip_reference_videos=True,
        hand_condition=False, camera_pose="latent-aligned first-relative W2C before logd4"), indent=2) + "\n")


if __name__ == "__main__":
    main()
