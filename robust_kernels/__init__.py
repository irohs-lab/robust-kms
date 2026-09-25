"""Matrix-free scalar and factored multioutput RKHS models."""

from .evaluate import kernel_eval_and_grad
from .fit import fit
from .state import initialize
from .objective import objective
from .updates import step

__all__ = ["kernel_eval_and_grad", "fit", "initialize", "objective", "step"]

from .multiclass_state import initialize as initialize_multiclass
from .multiclass_step import step as step_multiclass
from .multiclass_fit import fit as fit_multiclass
from .multioutput_evaluate import kernel_eval_and_grad as multioutput_eval_and_grad
from .rkhs_projection import project

__all__ += ["initialize_multiclass", "step_multiclass", "fit_multiclass",
            "multioutput_eval_and_grad", "project"]
