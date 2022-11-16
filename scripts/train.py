import src.trainer as trainer
import wandb


def main():
    wandb.init(project="DebateGPT")
    model = trainer.train()
    model.save("ckpts")


if __name__ == "__main__":
    main()
