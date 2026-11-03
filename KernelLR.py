import os
import time
import json
import torch
from functools import partial

from klr import KernelLogisticRegression
import dataset
from torchkernels.kernels.radial import laplacian, gaussian

import argparse


parser = argparse.ArgumentParser(
    description="Kernel Logistic Regression Hyperparameter Sweep"
)


parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=[
        "fmnist",
        "kmnist",
        "qmnist",
        "emnist",
        "svhn",
        "cifar"
    ],
    help="Dataset to use."
)


parser.add_argument(
    "--multiclass",
    action="store_true",
    help=(
        "Use multiclass classification. "
        "If this flag is absent, binary classification is used."
    )
)


parser.add_argument(
    "--classes",
    type=int,
    nargs="+",
    default=[0,1],
    help=(
        "Classes for binary experiments. "
        "Example: --classes 0 1 2 3 4 5 "
        "runs (0,1), (2,3), and (4,5)."
    )
)


parser.add_argument(
    "--kernel",
    type=str,
    required=True,
    choices=[
        "gaussian",
        "laplacian"
    ],
    help="Kernel to use."
)


args = parser.parse_args()

DATASET = args.dataset

MULTICLASS = args.multiclass

KERNEL_NAME = args.kernel

BATCH_SIZE = 256

# MAX_STEPS = 600 # For multiclass
MAX_STEPS = 200 # For binary

MIN_DELTA = 1e-6

LENGTH_SCALES = [
    0.25,
    # 0.5,
    # 1.0,
    # 2.0,
    3.0
    # 4.0
]

LEARNING_RATES = [
    0.001,
    0.01,
    # 0.1,
    # 0.5,
    # 1.0,
    # 2.0,
    # 5.0
]

if KERNEL_NAME == "gaussian":

    KERNEL_FUNCTION = gaussian

elif KERNEL_NAME == "laplacian":

    KERNEL_FUNCTION = laplacian

else:

    raise ValueError(
        f"Unsupported kernel: {KERNEL_NAME}"
    )

if MULTICLASS:

    CLASS_PAIRS = [
        (0, 1)
    ]

    NUM_CLASSES = 10


else:

    NUM_CLASSES = 2

    if args.classes is None:

        CLASS_PAIRS = [
            (0, 1)
        ]

    else:

        if len(args.classes) % 2 != 0:

            raise ValueError(
                "--classes must contain an even number "
                "of class labels.\n"
                "Example:\n"
                "--classes 0 1 2 3 4 5"
            )


        CLASS_PAIRS = [

            tuple(
                args.classes[i:i + 2]
            )

            for i in range(
                0,
                len(args.classes),
                2
            )
        ]

    for c0, c1 in CLASS_PAIRS:

        if c0 == c1:

            raise ValueError(
                f"Invalid class pair ({c0}, {c1}). "
                "The two classes must be different."
            )

def accuracy(
    model,
    X,
    y
):

    preds = model.predict(X)

    preds = torch.as_tensor(
        preds,
        device=y.device
    ).view(-1)

    return (
        (preds == y)
        .float()
        .mean()
        .item()
    )

def to_class_indices(y):

    if y.ndim > 1:

        return (
            y.argmax(dim=1)
            .long()
        )

    return (
        y.long()
        .view(-1)
    )


print("\n" + "#" * 80)

print(
    "KERNEL LOGISTIC REGRESSION EXPERIMENT"
)

print("#" * 80)

print(
    f"Dataset:     {DATASET}"
)

print(
    f"Multiclass:  {MULTICLASS}"
)

print(
    f"Kernel:      {KERNEL_NAME}"
)

print(
    f"Batch size:  {BATCH_SIZE}"
)

print(
    f"Max steps:   {MAX_STEPS}"
)


if MULTICLASS:

    print(
        "Classes:     (0, 1) "
        "[default/ignored in multiclass mode]"
    )

else:

    print(
        f"Class pairs: {CLASS_PAIRS}"
    )


print("#" * 80)

for CLASSES in CLASS_PAIRS:

    if MULTICLASS:

        OUTPUT_DIR = os.path.join(
            f"klr_sweep_{DATASET}",
            KERNEL_NAME
        )

    else:

        OUTPUT_DIR = os.path.join(
            f"klr_sweep_{DATASET}",
            KERNEL_NAME,
            f"binary_{CLASSES[0]}vs{CLASSES[1]}"
        )


    RUNS_DIR = os.path.join(
        OUTPUT_DIR,
        "runs"
    )


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    os.makedirs(
        RUNS_DIR,
        exist_ok=True
    )

    print("\n" + "#" * 80)


    if MULTICLASS:

        print(
            f"STARTING MULTICLASS EXPERIMENT"
        )

        print(
            f"Dataset: {DATASET}"
        )

    else:

        print(
            "STARTING BINARY EXPERIMENT"
        )

        print(
            f"Dataset: {DATASET}"
        )

        print(
            f"Classes: "
            f"{CLASSES[0]} vs {CLASSES[1]}"
        )


    print(
        f"Kernel: {KERNEL_NAME}"
    )

    print(
        f"Number of model classes: "
        f"{NUM_CLASSES}"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )

    print("#" * 80)

    if DATASET in [
        "fmnist",
        "kmnist",
        "qmnist",
        "emnist"
    ]:

        trainloader, valloader, testloader = (
            dataset.load_fmnist(

                classes=CLASSES,

                which_mnist=DATASET,

                multiclass=MULTICLASS,

                batch_size=BATCH_SIZE
            )
        )


    elif DATASET == "svhn":

        trainloader, valloader, testloader = (
            dataset.get_svhn(

                classes=CLASSES,

                multiclass=MULTICLASS,

                batch_size=BATCH_SIZE
            )
        )


    elif DATASET == "cifar":

        trainloader, valloader, testloader = (
            dataset.get_cifar(

                classes=CLASSES,

                multiclass=MULTICLASS,

                batch_size=BATCH_SIZE
            )
        )


    else:

        raise ValueError(
            f"Unsupported dataset: {DATASET}"
        )

    X_train, y_train = (
        trainloader.dataset.tensors
    )

    X_val, y_val = (
        valloader.dataset.tensors
    )

    X_test, y_test = (
        testloader.dataset.tensors
    )

    y_train = to_class_indices(
        y_train
    )

    y_val = to_class_indices(
        y_val
    )

    y_test = to_class_indices(
        y_test
    )

    print("\n" + "=" * 80)

    print(
        "DATASET INFORMATION"
    )

    print("=" * 80)


    print(
        "X_train shape:",
        X_train.shape
    )

    print(
        "y_train shape:",
        y_train.shape
    )


    print(
        "X_val shape:",
        X_val.shape
    )

    print(
        "y_val shape:",
        y_val.shape
    )


    print(
        "X_test shape:",
        X_test.shape
    )

    print(
        "y_test shape:",
        y_test.shape
    )


    print(
        "\nTrain class counts:",
        torch.unique(
            y_train,
            return_counts=True
        )
    )


    print(
        "Validation class counts:",
        torch.unique(
            y_val,
            return_counts=True
        )
    )


    print(
        "Test class counts:",
        torch.unique(
            y_test,
            return_counts=True
        )
    )


    print(
        "\nMinimum Y_train:",
        y_train.min().item()
    )

    print(
        "Maximum Y_train:",
        y_train.max().item()
    )


    print(
        "Minimum Y_val:",
        y_val.min().item()
    )

    print(
        "Maximum Y_val:",
        y_val.max().item()
    )


    print(
        "Minimum Y_test:",
        y_test.min().item()
    )

    print(
        "Maximum Y_test:",
        y_test.max().item()
    )

    best_model = None

    best_val_acc = -1.0

    best_length_scale = None

    best_learning_rate = None

    best_result = None


    total_runs = (
        len(LENGTH_SCALES)
        *
        len(LEARNING_RATES)
    )


    run_number = 0

    for length_scale in LENGTH_SCALES:

        for learning_rate in LEARNING_RATES:

            run_number += 1


            print(
                "\n" + "=" * 80
            )


            print(
                f"Run "
                f"{run_number}/"
                f"{total_runs}"
            )


            if MULTICLASS:

                print(
                    "Classification: "
                    "multiclass"
                )

            else:

                print(
                    "Classes: "
                    f"{CLASSES[0]} "
                    "vs "
                    f"{CLASSES[1]}"
                )


            print(
                f"Kernel: "
                f"{KERNEL_NAME}"
            )


            print(
                f"Length scale: "
                f"{length_scale}"
            )


            print(
                f"Learning rate: "
                f"{learning_rate}"
            )


            print(
                "=" * 80
            )

            model = (
                KernelLogisticRegression(

                    kernel=partial(
                        KERNEL_FUNCTION,
                        length_scale=length_scale,
                        in_place=False
                    ),

                    batch_size=BATCH_SIZE,

                    num_classes=NUM_CLASSES,

                    learning_rate=learning_rate,

                    max_steps=MAX_STEPS,

                    min_delta=MIN_DELTA
                )
            )


            start = (
                time.perf_counter()
            )


            model.fit(

                X_train,

                y_train,

                X_val,

                y_val
            )


            end = (
                time.perf_counter()
            )


            elapsed_time = (
                end - start
            )

            train_acc = accuracy(
                model,
                X_train,
                y_train
            )


            val_acc = accuracy(
                model,
                X_val,
                y_val
            )


            test_acc = accuracy(
                model,
                X_test,
                y_test
            )


            print(
                f"Train accuracy: "
                f"{train_acc * 100:.2f}%"
            )


            print(
                f"Val accuracy:   "
                f"{val_acc * 100:.2f}%"
            )


            print(
                f"Test accuracy:  "
                f"{test_acc * 100:.2f}%"
            )


            print(
                f"Time taken: "
                f"{elapsed_time:.3f} seconds"
            )

            run_result = {

                "run_number":
                    run_number,

                "dataset":
                    DATASET,

                "multiclass":
                    MULTICLASS,

                "classes":
                    (
                        [0, 1]
                        if MULTICLASS
                        else list(CLASSES)
                    ),

                "kernel":
                    KERNEL_NAME,

                "length_scale":
                    length_scale,

                "learning_rate":
                    learning_rate,

                "batch_size":
                    BATCH_SIZE,

                "num_classes":
                    NUM_CLASSES,

                "max_steps":
                    MAX_STEPS,

                "min_delta":
                    MIN_DELTA,

                "train_accuracy":
                    train_acc,

                "val_accuracy":
                    val_acc,

                "test_accuracy":
                    test_acc,

                "training_time_seconds":
                    elapsed_time,

                "X_train_shape":
                    list(
                        X_train.shape
                    ),

                "X_val_shape":
                    list(
                        X_val.shape
                    ),

                "X_test_shape":
                    list(
                        X_test.shape
                    ),

                "num_train_samples":
                    X_train.shape[0],

                "num_val_samples":
                    X_val.shape[0],

                "num_test_samples":
                    X_test.shape[0],
            }

            run_filename = (

                f"run_{run_number:03d}"

                f"_ls_{length_scale}"

                f"_lr_{learning_rate}"

                f".json"
            )


            run_path = os.path.join(
                RUNS_DIR,
                run_filename
            )


            with open(
                run_path,
                "w"
            ) as f:

                json.dump(
                    run_result,
                    f,
                    indent=4
                )


            print(
                "Run JSON saved to: "
                f"{run_path}"
            )

            if val_acc > best_val_acc:

                print(
                    "\nNEW BEST MODEL"
                )


                print(
                    "Validation accuracy: "
                    f"{val_acc * 100:.2f}%"
                )


                best_val_acc = (
                    val_acc
                )


                best_length_scale = (
                    length_scale
                )


                best_learning_rate = (
                    learning_rate
                )


                best_model = (
                    model
                )


                best_result = (
                    run_result
                )

    print(
        "\n" + "#" * 80
    )

    print(
        "BEST MODEL"
    )

    print(
        "#" * 80
    )


    print(
        f"Dataset: "
        f"{DATASET}"
    )


    if MULTICLASS:

        print(
            "Classification: "
            "Multiclass"
        )

    else:

        print(
            f"Classes: "
            f"{CLASSES[0]} "
            f"vs "
            f"{CLASSES[1]}"
        )


    print(
        f"Kernel: "
        f"{KERNEL_NAME}"
    )


    print(
        f"Length scale: "
        f"{best_length_scale}"
    )


    print(
        f"Learning rate: "
        f"{best_learning_rate}"
    )


    print(
        f"Train accuracy: "
        f"{best_result['train_accuracy'] * 100:.2f}%"
    )


    print(
        f"Validation accuracy: "
        f"{best_result['val_accuracy'] * 100:.2f}%"
    )


    print(
        f"Test accuracy: "
        f"{best_result['test_accuracy'] * 100:.2f}%"
    )

    best_model_path = os.path.join(
        OUTPUT_DIR,
        "best_klr_model.pt"
    )


    torch.save(

        {

            "model":
                best_model,

            "model_config": {

                "kernel":
                    KERNEL_NAME,

                "length_scale":
                    best_length_scale,

                "learning_rate":
                    best_learning_rate,

                "batch_size":
                    BATCH_SIZE,

                "num_classes":
                    NUM_CLASSES,

                "max_steps":
                    MAX_STEPS,

                "min_delta":
                    MIN_DELTA,
            },

            "dataset_config": {

                "dataset":
                    DATASET,

                "multiclass":
                    MULTICLASS,

                "classes":
                    (
                        [0, 1]
                        if MULTICLASS
                        else list(CLASSES)
                    ),
            },

            "metrics":
                best_result,

        },

        best_model_path
    )


    print(
        f"\nBest model saved to: "
        f"{best_model_path}"
    )

    best_json_path = os.path.join(
        OUTPUT_DIR,
        "best_run.json"
    )


    with open(
        best_json_path,
        "w"
    ) as f:

        json.dump(
            best_result,
            f,
            indent=4
        )


    print(
        f"Best run JSON saved to: "
        f"{best_json_path}"
    )


    print(
        "\n" + "#" * 80
    )


    if MULTICLASS:

        print(
            "FINISHED MULTICLASS EXPERIMENT"
        )

    else:

        print(
            f"FINISHED CLASS PAIR: "
            f"{CLASSES[0]} "
            f"vs "
            f"{CLASSES[1]}"
        )


    print(
        "#" * 80
    )

print(
    "\n" + "=" * 80
)

print(
    "ALL EXPERIMENTS FINISHED"
)

print(
    "=" * 80
)