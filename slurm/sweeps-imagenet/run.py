import subprocess
import textwrap

def submit_job(args, debug: bool = False):
    script = f"""#!/bin/bash
#SBATCH --job-name=ssl-imagenet
#SBATCH --output=/work/FAC/FBM/DBC/mrapsoma/prometex/logs/adrianom/ssl-imagenet-%j.log
#SBATCH --error=/work/FAC/FBM/DBC/mrapsoma/prometex/logs/adrianom/ssl-imagenet-%j.err

#SBATCH --account=mrapsoma_prometex

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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

sweep = []
for teacher_temp in [0.06, 0.08]:
    sweep.extend([
        # DINOv1, resnet18
        # Baseline
        # f"clis/dinov1.py fit --config configs/dinov1-imagenet-resnet.yaml --model.dino_head_output_dim=2048 --model.dino_head_hidden_dim=512 --model.dino_head_bottleneck_dim=64 --model.teacher_temp={teacher_temp}",

        f"clis/dinov1.py fit --config configs/dinov1-imagenet-resnet.yaml --model.dino_head_output_dim=512 --model.dino_head_hidden_dim=256 --model.dino_head_bottleneck_dim=16 --model.teacher_temp={teacher_temp} --model.lr=0.007",
        f"clis/dinov1.py fit --config configs/dinov1-imagenet-resnet.yaml --model.dino_head_output_dim=1024 --model.dino_head_hidden_dim=512 --model.dino_head_bottleneck_dim=32 --model.teacher_temp={teacher_temp} --model.lr=0.005",
        f"clis/dinov1.py fit --config configs/dinov1-imagenet-resnet.yaml --model.dino_head_output_dim=4096 --model.dino_head_hidden_dim=2048 --model.dino_head_bottleneck_dim=256 --model.teacher_temp={teacher_temp}  --model.lr=0.003",
        f"clis/dinov1.py fit --config configs/dinov1-imagenet-resnet.yaml --model.dino_head_output_dim=8192 --model.dino_head_hidden_dim=4096 --model.dino_head_bottleneck_dim=512 --model.teacher_temp={teacher_temp} --model.lr=0.001",

        # MAEv1 (optional)
        # "clis/maev1.py fit --config configs/maev1-imagenet-vit.yaml",
    ])

debug = False
# debug = True
for args in sweep:
    if debug:
        args += ' --trainer.fast_dev_run=true'
    submit_job(args=args, debug=debug)
