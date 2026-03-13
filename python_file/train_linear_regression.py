#!/usr/bin/env python3

import sys
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LinearRegression


def main():
    if len(sys.argv) != 3:
        print("Usage: python train_linear_regression.py <input_libsvm_file> <output_model_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Loading data from: {input_file}")

    # Load LIBSVM format data
    X, y = load_svmlight_file(input_file)

    print(f"Data shape: {X.shape}")
    print("Training Linear Regression model...")

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    weights = model.coef_
    bias = model.intercept_

    print(f"Number of features: {len(weights)}")
    print(f"Bias: {bias}")

    print(f"Saving model to: {output_file}")

    # Save weights in LIBLINEAR-like format
    with open(output_file, "w") as f:
        f.write("solver_type SKLEARN_LINEAR_REGRESSION\n")
        f.write("nr_feature {}\n".format(len(weights)))
        f.write("bias {}\n".format(bias))
        f.write("w\n")

        for w in weights:
            f.write(str(w) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()