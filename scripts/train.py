import debategpt.training.trainer as trainer
import wandb


def main():
    model = trainer.train()
    model.save_pretrained("ckpts")


if __name__ == "__main__":
    main()
