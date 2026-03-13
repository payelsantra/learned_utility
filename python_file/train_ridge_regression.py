#!/usr/bin/env python3

import sys
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import Ridge


def main():
    if len(sys.argv) != 3:
        print("Usage: python train_ridge_regression.py <input_libsvm_file> <output_model_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Loading data from: {input_file}")

    # Load LIBSVM format data
    X, y = load_svmlight_file(input_file)

    print(f"Data shape: {X.shape}")
    print("Training Ridge Regression model...")

    # Regularization strength
    alpha = 1.0

    # Train Ridge model
    model = Ridge(alpha=alpha, fit_intercept=True)

    model.fit(X, y)

    weights = model.coef_
    bias = model.intercept_

    print(f"Number of features: {len(weights)}")
    print(f"Bias: {bias}")
    print(f"Alpha (regularization): {alpha}")

    print(f"Saving model to: {output_file}")

    # Save weights in LIBLINEAR-like format
    with open(output_file, "w") as f:
        f.write("solver_type SKLEARN_RIDGE_REGRESSION\n")
        f.write("nr_feature {}\n".format(len(weights)))
        f.write("bias {}\n".format(bias))
        f.write("alpha {}\n".format(alpha))
        f.write("w\n")

        for w in weights:
            f.write(str(w) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()

    # python3 train_ridge_regression.py /media/pbclab/One_Touch_july_2/QPP/optim_RAG/code/learned_sparse/models/termwts_minmax.txt /media/pbclab/One_Touch_july_2/QPP/optim_RAG/code/learned_sparse/models/model_ridge_input_Wminmax.txt