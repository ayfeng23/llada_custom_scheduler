To initialize our conda environment to run the finetuning and testing scripts, we have an ```environment.yml``` file for convenience. 

To run our LLaDA baseline finetuning, run ```LLaDA/train_llada_lora_sched_masking.py --baseline --weighted```. If you are on an HPC, you can run our slurm script using ```sbatch llada_baseline.slurm```. This script will run LoRA finetuning for the baseline model using the uniform masking.

To run our LLaDA scheduled masking finetuning, run ```LLaDA/train_llada_lora_sched_masking.py --weighted```. If you are on an HPC, you can run our slurm script using ```sbatch llada_sched.slurm```. This script will run LoRA finetuning for the updated model using the custom masking scheduler.

After running the LoRA finetuning, your weights will be saved in a folder to your directory, along with frequent validation testing printed out to your console or log file.

To run the final testing script, you can run ```LLaDA/run_testing.py```. If you are on an HPC, you can run our slurm script using ```sbatch run_testing.slurm```. This script will run our testing script for the pretrained baseline model, the finetuned baseline model, and the finetuned masking scheduler model.
