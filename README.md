# Assignment 5 - DLOps

## Q1: ViT with LoRA on CIFAR-100
- Fine-tuned ViT-S on CIFAR-100 with and without LoRA using PEFT
- LoRA injected into Q, K, V attention weights
- Hyperparameter search using Optuna

See [q1/README.md](q1/README.md) for setup and results.

#Links
- WandB: https://wandb.ai/sandeshsuman2000-iit-jodhpur/m25csa034_MLDLops
- HuggingFace: https://huggingface.co/sandesh2233/vit-lora-cifar100

## Q2: Adversarial Attacks using IBM ART
- Trained ResNet-18 on CIFAR-10 from scratch (Best Accuracy: 85.46%)
- FGSM attack implemented from scratch and via IBM ART
- Adversarial detection using ResNet-34 for PGD and BIM attacks

See [q2/README.md](q2/README.md) for setup and results.
The best model is saved in huggingface for Q2

## Links
- WandB Q2: https://wandb.ai/sandeshsuman2000-iit-jodhpur/DLOps-Ass5-Q2
- HuggingFace Q2: https://huggingface.co/sandesh2233/dlops-ass5-q2
