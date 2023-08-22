from os.path import join

TEXTURE_SETS_DIR = join("experiments", "texture_sets")
DATASETS_DIR = join("experiments", "datasets")
MODELS_DIR = join("experiments", "models")
AFF_CHOICES = ["SA1", "RA", "PC", "all"]

PERCENT_TRAINING = 70
PERCENT_VALIDATION = 15
PERCENT_TEST = 15
assert PERCENT_TRAINING + PERCENT_VALIDATION + PERCENT_TEST == 100

SEED = 2027