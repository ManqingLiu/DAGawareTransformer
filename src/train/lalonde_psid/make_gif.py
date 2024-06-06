import os

import wandb

from src.train.lalonde_psid.train_metrics import images_to_gif


def main(wandb_run_path: str, duration: int) -> None:
    api = wandb.Api()
    run = api.run(wandb_run_path)
    for file in run.files():
        if file.name.endswith(".png") and "Train" in file.name:
            file.download(root="train_gif_images", replace=True, exist_ok=True)
        if file.name.endswith(".png") and "Val" in file.name:
            file.download(root="val_gif_images", replace=True, exist_ok=True)

    for split, filepath in {
        "train": "train_gif_images",
        "val": "val_gif_images",
    }.items():
        image_filepaths = []
        image_directory = os.path.join(filepath, "media/images")
        for root, dirs, files in os.walk(image_directory):
            for file in files:
                if file.endswith(".png"):
                    image_filepaths.append(os.path.join(root, file))

        images_to_gif(
            image_filepaths,
            gif_outpath=f"experiments/results/figures/{split}_propensity_score.gif",
            duration=duration,
        )

        for root, dirs, files in os.walk(image_directory):
            for file in files:
                os.remove(os.path.join(root, file))


if __name__ == "__main__":
    main('mliu7/DAG transformer/09semqwp', 500)
