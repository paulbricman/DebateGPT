import src.trainer as trainer
import wandb


def main():
    wandb.init(project="DebateGPT")
    model = trainer.train()
    model.save(".", "DebateGPT")


if __name__ == "__main__":
    main()
