import itertools
import subprocess
import shlex
import os
import json
from pathlib import Path

# datasets = ["adult","bank", "phishing"]
datasets = ["adult"]
# models = ["mlp"]
models = ["rfm"] # On mercer. Only laplacian for now.
# models = ["rfm"]

if models[0]=="mlp":
    model_ckpts = {
        ("adult", "mlp"): [
            "trained-models/adult-mlp.ckpt",
            # "/path/to/another/adult_mlp_seed1.ckpt",
            # "/path/to/another/adult_mlp_advtrained.ckpt",
        ],
        ("bank", "mlp"): [
            "trained-models/bank-mlp.ckpt",
            # "/path/to/another/adult_mlp_seed1.ckpt",
            # "/path/to/another/adult_mlp_advtrained.ckpt",
        ],
        ("phishing", "mlp"): [
            "trained-models/phishing-mlp.ckpt",
        ]
    }
elif models[0] == "kernel":
    model_ckpts = {
        ("adult", "kernel"): [
            "trained-models/kernel/adult-laplacian-kernel.pt",
            # "/path/to/another/adult_mlp_seed1.ckpt",
            # "/path/to/another/adult_mlp_advtrained.ckpt",
        ],
        ("bank", "kernel"): [
            "trained-models/kernel/bank-laplacian-kernel.pt",
            # "/path/to/another/adult_mlp_seed1.ckpt",
            # "/path/to/another/adult_mlp_advtrained.ckpt",
        ],
        ("phishing", "kernel"): [
            "trained-models/kernel/phishing-laplacian-kernel.pt",
            # "/path/to/another/adult_mlp_seed1.ckpt",
            # "/path/to/another/adult_mlp_advtrained.ckpt",
        ]
    }
elif models[0] == "rfm":
    model_ckpts = {
        ("adult", "rfm"): [
            "trained-models/kernel/adult-laplacian-rfm.pt",
            # "/path/to/another/adult_mlp_seed1.ckpt",
            # "/path/to/another/adult_mlp_advtrained.ckpt",
        ],
        ("bank", "rfm"): [
            "trained-models/kernel/bank-laplacian-rfm.pt",
            # "/path/to/another/adult_mlp_seed1.ckpt",
            # "/path/to/another/adult_mlp_advtrained.ckpt",
        ],
        ("phishing", "rfm"): [
            "trained-models/kernel/phishing-laplacian-rfm.pt",
            # "/path/to/another/adult_mlp_seed1.ckpt",
            # "/path/to/another/adult_mlp_advtrained.ckpt",
        ]
    }

# eps_values = [0.01, 0.03333333333333333 , 0.075]
eps_values = [0.075]

seeds = [0]

max_iter_tabpgd = 100
max_iter_cafa = 250
perturb_cat_steps = 10

base_out = "./laplacian_ker_adult_075"

# if models[0] == "mlp":
#     base_out = "sweep_outputs_mlp_bochner"
# elif models[0] == "kernel":
#     base_out = "sweep_outputs_kernel_new_bochner"
# elif models[0] == "rfm":
#     base_out = "sweep_outputs_rfm_new_bochner"
 
os.makedirs(base_out, exist_ok=True)

runs = []

for data, model, eps, seed in itertools.product(datasets, models, eps_values, seeds):

    ckpt_list = model_ckpts[(data, model)]

    for model_path in ckpt_list:
        step_size = eps / 100.0
        ckpt_name = Path(model_path).stem

        run_name = (
            f"data-{data}_model-{model}_ckpt-{ckpt_name}_"
            f"eps-{eps}_step-{step_size}_seed-{seed}"
        ).replace("/", "_")

        out_dir = os.path.join(base_out, run_name)

        cmd = [
            "python", "attack.py",

            f"data={data}",
            f"ml_model={model}",

            # important checkpoint overrides
            "ml_model.perform_training=False",
            "ml_model.perform_grid_search_hparams=False",
            f"ml_model.model_artifact_path={model_path}",

            "attack=cafa",
            "constraints=dcs",

            f"attack.eps={eps}",
            f"attack.step_size={step_size}",
            f"attack.random_seed={seed}",
            f"attack.max_iter={max_iter_cafa}",
            f"attack.max_iter_tabpgd={max_iter_tabpgd}",
            f"attack.perturb_categorical_each_steps={perturb_cat_steps}",

            f"constraints.constrainer_params.cost_ball_eps={eps}",
            "constraints.constrainer_params.limit_cost_ball=True",
            "constraints.projector_params.upper_projection_budget_bound=0.5",

            "perform_attack=True",
            "perform_projection=True",
            "perform_constraints_mining=False",
            "perform_constraints_ranking=False",
            "perform_constraints_soundness_evaluation=False",

            "data_split_to_attack=test",
            "n_samples_to_attack=1500",

            f"hydra.run.dir={out_dir}",
            f"hydra.job.name={run_name}",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""

        result = subprocess.run(
            cmd,
            env=env,
        )

        print("\nRunning:")
        print(" ".join(shlex.quote(x) for x in cmd))

        # result = subprocess.run(cmd)

        runs.append({
            "data": data,
            "model": model,
            "model_path": model_path,
            "eps": eps,
            "step_size": step_size,
            "seed": seed,
            "out_dir": out_dir,
            "returncode": result.returncode,
        })

        with open(os.path.join(base_out, "sweep_log.json"), "w") as f:
            json.dump(runs, f, indent=2)

print("\nSweep finished.")
print("Log saved to:", os.path.join(base_out, "sweep_log.json"))