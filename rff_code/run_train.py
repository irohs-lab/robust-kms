import os
import argparse
import torch

from data_loading import (
    get_class_pairs,
    load_data_for_experiment,
    loader_to_tensors,
)

from configs import (
    SQUARE_LAMDAS,
    CE_WEIGHT_DECAYS,
    CE_LEARNING_RATES,
    ADV_LEARNING_RATES,
    LENGTH_SCALES,
    LENGTH_SCALE,
    CUML_CS,
    ADV_CONFIGS,
    get_num_classes,
    get_num_features,
)

from save_utils import (
    get_output_dir,
    get_save_names,
    save_square_result,
    save_ce_result,
    save_cuml_ce_result,
)

from trainers.ce_torch import fit_rff_cross_entropy, fit_rff_cross_entropy_cv
from trainers.ce_cuml import fit_rff_cross_entropy_cuml, fit_rff_cross_entropy_cuml_cv
from trainers.adv_ce import fit_rff_adv_cross_entropy
from trainers.ridge import fit_rff_ridge, fit_rff_ridge_cv


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--MC", action="store_true", help="Run multiclass instead of pairwise")
    parser.add_argument("--DATASET", type=str, default="kmnist")
    parser.add_argument("--kernel", type=str, default="Laplace")
    parser.add_argument(
        "--trn",
        type=str,
        choices=[
            "square",
            "square_cv",
            "ce",
            "ce_torch_cv",
            "adv_ce",
            "ce_cuml",
            "ce_cuml_cv",
        ],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num_features", type=int, default=15000, help="Number of random features to use")
    parser.add_argument("--CUML_CS", type=float,nargs="+", help="List of C values for cuML logistic regression")
    parser.add_argument("--LENGTH_SCALE", type=float, nargs="+", help="List of length scales")
    parser.add_argument("--SQUARE_LAMDAS", type=float, nargs="+", help="List of lambda values for ridge regression")

    return parser.parse_args()

def run_one_experiment(
    Dataset,
    kernel_type,
    training_type,
    mc,
    cl1,
    cl2,
    seed,
    device,
    num_features,
    CUML_CS=None,
    LENGTH_SCALE=None,
    SQUARE_LAMDAS=None,
):
    classes = (cl1, cl2)

    trainloader, valloader, testloader = load_data_for_experiment(
        dataset_name=Dataset,
        cl1=cl1,
        cl2=cl2,
        multiclass=mc,
    )

    for x, y in trainloader:
        print("Batch x shape:", x.shape)
        print("Batch y shape:", y.shape)
        break

    num_classes = get_num_classes(mc)
    num_features = get_num_features(mc, num_features)

    print("Number of classes:", num_classes)
    print("Number of features:", num_features)

    x_train, y_train = loader_to_tensors(trainloader, device, flatten=True)
    x_val, y_val = loader_to_tensors(valloader, device, flatten=True)
    x_test, y_test = loader_to_tensors(testloader, device, flatten=True)

    if training_type == "square":
        best, all_results = fit_rff_ridge(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            num_classes=num_classes,
            num_features=num_features,
            kernel_type=kernel_type,
            lamdas=SQUARE_LAMDAS,
            length_scales=LENGTH_SCALE,
            seed=seed,
            device=device,
        )

    elif training_type == "ce":
        best, all_results = fit_rff_cross_entropy(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            num_classes=num_classes,
            num_features=num_features,
            kernel_type=kernel_type,
            weight_decays=CE_WEIGHT_DECAYS,
            learning_rates=CE_LEARNING_RATES,
            length_scales=LENGTH_SCALES,
            epochs=300,
            seed=seed,
            device=device,
        )

    elif training_type == "ce_torch_cv":
        best, all_results = fit_rff_cross_entropy_cv(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            num_classes=num_classes,
            num_features=num_features,
            kernel_type=kernel_type,
            weight_decays=CE_WEIGHT_DECAYS,
            learning_rates=CE_LEARNING_RATES,
            length_scales=LENGTH_SCALES,
            epochs=500,
            n_splits=5,
            seed=seed,
            device=device,
        )

    elif training_type == "square_cv":
        best, all_results = fit_rff_ridge_cv(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            num_classes=num_classes,
            num_features=num_features,
            kernel_type=kernel_type,
            lamdas=SQUARE_LAMDAS,
            length_scales=LENGTH_SCALES,
            n_splits=5,
            seed=seed,
            device=device,
        )

    elif training_type == "ce_cuml":
        best, all_results = fit_rff_cross_entropy_cuml(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            num_classes=num_classes,
            num_features=num_features,
            kernel_type=kernel_type,
            Cs=CUML_CS,
            length_scales=LENGTH_SCALE,
            max_iter=5000,
            seed=seed,
            device=device,
        )
    elif training_type == "ce_cuml_cv":
        best, all_results = fit_rff_cross_entropy_cuml_cv(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            num_classes=num_classes,
            num_features=num_features,
            kernel_type=kernel_type,
            Cs=CUML_CS,
            length_scales=LENGTH_SCALES,
            max_iter=5000,
            n_splits=5,
            seed=seed,
            device=device,
        )
    elif training_type == "adv_ce":
        adv_gen = fit_rff_adv_cross_entropy(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            num_classes=num_classes,
            num_features=num_features,
            kernel_type=kernel_type,
            weight_decays=CE_WEIGHT_DECAYS,
            learning_rates=ADV_LEARNING_RATES,
            length_scales=LENGTH_SCALES,
            epochs=50,
            adv_configs=ADV_CONFIGS,
            batch_size=256,
            adv_steps=20,
            seed=seed,
            device=device,
        )

        return adv_gen, num_classes, classes

    else:
        raise ValueError(f"Unsupported training_type: {training_type}")

    return best, all_results, num_classes, classes

def main():
    args = parse_args()

    Dataset = args.DATASET.lower()
    kernel_type = args.kernel.lower()
    mc = args.MC
    training_type = args.trn
    seed = args.seed
    num_features = args.num_features
    CUML_CS = args.CUML_CS
    LENGTH_SCALE = args.LENGTH_SCALE
    SQUARE_LAMDAS = args.SQUARE_LAMDAS

    print("CUML_CS:", CUML_CS)
    print(type(CUML_CS), "LENGTH_SCALE:", LENGTH_SCALE, type(LENGTH_SCALE))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Dataset:", Dataset)
    print("Kernel:", kernel_type)
    print("Training:", training_type)
    print("MC:", mc)
    print("SEED:", seed)
    print("Device:", device)
    print("Number of features:", num_features)

    classes1, classes2 = get_class_pairs(Dataset, mc)

    out_dir = get_output_dir(
        kernel_type=kernel_type,
        dataset_name=Dataset,
        multiclass=mc,
        training_type=training_type,
        seed=seed
    )

    os.makedirs(out_dir, exist_ok=True)

    for cl1 in classes1:
        for cl2 in range(cl1 + 1, len(classes2) + 1):
            classes = (cl1, cl2)

            result = run_one_experiment(
                Dataset=Dataset,
                kernel_type=kernel_type,
                training_type=training_type,
                mc=mc,
                cl1=cl1,
                cl2=cl2,
                seed=seed,
                device=device,
                num_features=num_features,
                CUML_CS=CUML_CS,
                LENGTH_SCALE=LENGTH_SCALE,
                SQUARE_LAMDAS=SQUARE_LAMDAS
            )

            if training_type == "adv_ce":
                adv_gen, num_classes, classes = result

                for adv_folder, best, all_results in adv_gen:
                    print(f"Best adv result for {adv_folder}:")
                    print(best)

                continue

            best, all_results, num_classes, classes = result

            save_name, save_ckpt = get_save_names(
                dataset_name=Dataset,
                classes=classes,
                multiclass=mc,
                seed=seed,
                num_features=get_num_features(multiclass=mc, num_features=num_features)
            )

            if training_type in ["square", "square_cv"]:
                save_square_result(
                    out_dir=out_dir,
                    save_name=save_name,
                    save_ckpt=save_ckpt,
                    best=best,
                    all_results=all_results,
                    num_classes=num_classes,
                )

            elif training_type in ["ce", "ce_torch_cv"]:
                save_ce_result(
                    out_dir=out_dir,
                    save_name=save_name,
                    save_ckpt=save_ckpt,
                    best=best,
                    all_results=all_results,
                    num_classes=num_classes,
                )

            elif training_type in ["ce_cuml", "ce_cuml_cv"]:
                save_cuml_ce_result(
                    out_dir=out_dir,
                    save_name=save_name,
                    save_ckpt=save_ckpt,
                    best=best,
                    all_results=all_results,
                    num_classes=num_classes,
                )

if __name__ == "__main__":
    main()