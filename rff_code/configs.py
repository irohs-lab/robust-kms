SQUARE_LAMDAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


CE_WEIGHT_DECAYS = [1e-4, 1e-3, 1e-2, 1e-1]
# CE_WEIGHT_DECAYS = [1e-4] # Retrain to 20k epochs


# CE_LEARNING_RATES = [0.001, 0.01, 0.03, 0.1]
CE_LEARNING_RATES = [0.01] # Retrain to 20k epochs


ADV_LEARNING_RATES = [0.02, 0.1]

# LENGTH_SCALES = [0.1, 0.3, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
LENGTH_SCALES = [0.3, 0.5, 0.75, 1.0, 2.0, 3.0]
# LENGTH_SCALES = [0.5, 0.75, 1.0, 2.0, 3.0]

LENGTH_SCALE = [0.5]

# LENGTH_SCALES = [0.75, 1.0]# Retrain to 20k epochs

# CUML_CS = [1e7, 1e6, 1e5, 1e4, 1e3, 1e2]
CUML_CS = [1e2]


ADV_CONFIGS = (
    {"norm": "Linf", "eps": 6 / 255, "alpha": 6 / 255 / 4, "folder": "linf_6"},
    {"norm": "L2", "eps": 1.0, "alpha": 0.25, "folder": "l2_1.0"},
)


def get_num_features(multiclass,num_features):
    if multiclass:
        return num_features
    # return 1000
    return num_features


def get_num_classes(multiclass):
    if multiclass:
        return 10
    return 2