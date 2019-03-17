# Random seed used for all "random" generators
random_state = 1

# Number of parallel jobs used for some sklearn routines (-1 means use all cpus)
n_jobs = -1

# Fractional size of test dataset
test_size = 1/5.

# # Number of folds for GridSearchCV
# nfolds = 5

# Settings to control which features are used
pRows = 28
pCols = 28
pColNames = list(range(pRows * pCols))
pScale = 255.0 # Scale factor for pixels.  Pixel raw data is 0 to 255.  Set to 1 to avoid scaling

# Number of samples PER CATEGORY (defined below)
nSamples = 2500

# Names of all models to include
# names = ['baseball', 'basketball', 'camel', 'cow', 'clock', 'wristwatch']
names = ['baseball', 'basketball']

# Plot min/max extents for all accuracy plots
accuracyMin = 0.6
accuracyMax = 1.0

# Maximum iterations for iterative classification trainers
max_iter = 3000