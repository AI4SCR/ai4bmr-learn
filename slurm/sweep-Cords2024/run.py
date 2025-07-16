import subprocess
import textwrap

def submit_job(args, debug: bool = False):
    script = f"""#!/bin/bash
#SBATCH --job-name=clf-Cords2024
#SBATCH --output=/work/FAC/FBM/DBC/mrapsoma/prometex/logs/adrianom/clf-Cords2024-%j.log
#SBATCH --error=/work/FAC/FBM/DBC/mrapsoma/prometex/logs/adrianom/clf-Cords2024-%j.err

#SBATCH --account=mrapsoma_prometex

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Load environment
source /users/amarti51/miniconda3/bin/activate
conda activate beat

cd /work/FAC/FBM/DBC/mrapsoma/prometex/projects/ai4bmr-learn || exit

python {args}
"""
    script = textwrap.dedent(script)

    if debug:
        print(script)
        return

    proc = subprocess.run(
        ["sbatch"],
        input=script,
        text=True,
        check=True,
        capture_output=True,
    )
    job_id = proc.stdout.strip()
    print(f"Submitted array job {job_id}")


sweep = [
    # resnet18
    # "cli/clf.py --config configs/Cords2024/clf-cords2024-resnet.yaml --model.backbone.init_args.pretrained=false --model.freeze_backbone=true",
    # "cli/clf.py --config configs/Cords2024/clf-cords2024-resnet.yaml --model.backbone.init_args.pretrained=true --model.freeze_backbone=true",
    # "cli/clf.py --config configs/Cords2024/clf-cords2024-resnet.yaml --model.backbone.init_args.pretrained=true --model.freeze_backbone=false",
    # resnet50
    # "cli/clf.py fit --config configs/Cords2024/clf-cords2024-resnet.yaml --model.backbone.init_args.pretrained=false --model.freeze_backbone=true --model.backbone.init_args.model_name=resnet50 --model.input_dim=2048",
    # "cli/clf.py fit --config configs/Cords2024/clf-cords2024-resnet.yaml --model.backbone.init_args.pretrained=true --model.freeze_backbone=true --model.backbone.init_args.model_name=resnet50 --model.input_dim=2048",
    # "cli/clf.py fit --config configs/Cords2024/clf-cords2024-resnet.yaml --model.backbone.init_args.pretrained=true --model.freeze_backbone=false --model.backbone.init_args.model_name=resnet50 --model.input_dim=2048",
    # vit
    # "cli/clf.py fit --config configs/Cords2024/clf-cords2024-vit.yaml --model.backbone.init_args.pretrained=false --model.freeze_backbone=true",
    # "cli/clf.py fit --config configs/Cords2024/clf-cords2024-vit.yaml --model.backbone.init_args.pretrained=true --model.freeze_backbone=true",
    # "cli/clf.py fit --config configs/Cords2024/clf-cords2024-vit.yaml --model.backbone.init_args.pretrained=true --model.freeze_backbone=false",
    # GNN
    # "cli/clf.py fit --config configs/Cords2024/clf-cords2024-graph.yaml --model.freeze_backbone=false",
    # DINOv1
    # "clis/dinov1.py fit --config configs/Cords2024/dinov1-cords2024-vit.yaml",
    "clis/dinov1.py fit --config configs/Cords2024/dinov1-cords2024-resnet.yaml",
    # MAEv1
    "clis/maev1.py fit --config configs/Cords2024/maev1-cords2024-vit.yaml",
]

debug = True
for args in sweep:
    if debug:
        args += ' --trainer.fast_dev_run=true'
    submit_job(args=args, debug=debug)
