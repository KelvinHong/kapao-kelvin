import subprocess

validation_without_rect = [
    "python",
    "val.py",
    "--data",
    "data/coco-kp-light.yaml",
    "--batch-size",
    "4",
]
training_without_rect = [
    "python",
    "train.py",
    "--img",
    "1280",
    "--batch",
    "8",
    "--epochs",
    "1",
    "--data",
    "data/coco-kp-light.yaml",
    "--hyp",
    "data/hyps/hyp.kp-p6.yaml",
    "--val-scales",
    "1",
    "--val-flips",
    "-1",
    "--weights",
    "yolov5s6.pt",
    "--project",
    "runs/s_e500",
    "--name",
    "train",
    "--workers",
    "1",
]
validation_with_rect = validation_without_rect + ["--rect"]
training_with_rect = training_without_rect + ["--rect"]

remove_cache = ["rm", "-f", "data/datasets/coco/kp_labels/img_txt/*_mini.cache"]
cleanup = ["rm", "-rf", "runs"]


def run_command(command):
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f"Command '{' '.join(command)}' executed successfully.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing command '{' '.join(command)}':")
        raise


if __name__ == "__main__":
    for command in [
        training_without_rect,
        training_with_rect,
        validation_without_rect,
        validation_with_rect,
    ]:
        run_command(remove_cache)
        for _ in range(2):
            run_command(command)

    run_command(cleanup)
