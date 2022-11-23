import src.trainer as trainer
import wandb


def main():
    model = trainer.train()
    model.save("ckpts")


if __name__ == "__main__":
    main()
