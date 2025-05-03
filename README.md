**Environment and Setup**

To initialize our conda environment to run the finetuning and testing scripts, we have an ```environment.yml``` file for convenience, which contains all the external libraries and dependencies we used for finetuning as well as their versions. 

To run our LLaDA baseline finetuning, run ```LLaDA/train_llada_lora_sched_masking.py --baseline --weighted```. If you are on an HPC, you can run our slurm script using ```sbatch llada_baseline.slurm```. This script will run LoRA finetuning for the baseline model using the uniform masking.

To run our LLaDA scheduled masking finetuning, run ```LLaDA/train_llada_lora_sched_masking.py --weighted```. If you are on an HPC, you can run our slurm script using ```sbatch llada_sched.slurm```. This script will run LoRA finetuning for the updated model using the custom masking scheduler.

After running the LoRA finetuning, your weights will be saved in a folder to your directory, along with frequent validation testing printed out to your console or log file.

To run the final testing script, you can run ```LLaDA/run_testing.py```. If you are on an HPC, you can run our slurm script using ```sbatch run_testing.slurm```. This script will run our testing script for the pretrained baseline model, the finetuned baseline model, and the finetuned masking scheduler model.

**Compute Infrastructure**

We fine-tuned and performed inference with a single NVIDIA A100 GPU on Yale’s Grace cluster. Jobs were submitted via SLURM, requesting one A100 GPU, 4 CPU cores, and 32 GB of RAM for up to 4 hours per run (--gres=gpu:a100:1, --cpus-per-task=4, --mem=32G, --time=04:00:00). Fine-tuning was accelerated with bfloat16 mixed precision and TF32 matrix math.

**Dataset**

We use the [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) SFT dataset and filter for samples that contain at least one user-assistant response, and take the first such pair among each sample. Due to compute and time constraints, we use a subset of 60 thousand random samples to finetune our model. 

**Presentation and Report**

Please see our [slides](https://drive.google.com/file/d/1AsE3NJiMB6xW9ke8HFScwsLawjpfZkqC/view?usp=sharing) and [report](https://drive.google.com/file/d/1UdWmG3YBmZcJXTBxXCfwW2K8CfQwcRjo/view?usp=sharing).
