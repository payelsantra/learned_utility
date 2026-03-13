#!/usr/bin/env python3

import sys
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression


def main():

    if len(sys.argv) != 3:
        print("Usage: python train_infonce_logistic.py <input_libsvm_file> <output_model_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print("Loading data from:", input_file)

    # load sparse features
    X, y = load_svmlight_file(input_file)

    print("Data shape:", X.shape)
    print("Training logistic regression (InfoNCE approximation)...")

    # logistic regression works directly with sparse CSR matrices
    model = LogisticRegression(
        penalty="l2",
        solver="saga",
        max_iter=200,
        n_jobs=-1
    )

    model.fit(X, y)

    weights = model.coef_[0]
    bias = model.intercept_[0]

    print("Number of features:", len(weights))
    print("Bias:", bias)

    print("Saving model to:", output_file)

    # Save in same format as previous regression models
    with open(output_file, "w") as f:

        f.write("solver_type SKLEARN_LOGISTIC_INFO_NCE\n")
        f.write("nr_feature {}\n".format(len(weights)))
        f.write("bias {}\n".format(bias))
        f.write("w\n")

        for w in weights:
            f.write(str(w) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()