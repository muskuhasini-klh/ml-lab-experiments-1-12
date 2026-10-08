import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add src/data to Python path
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data"
    )
)

# Import data ingestion function
from ingest import load_and_validate_data


def train_linear_regression_ls():

    # ==========================================================
    # 1. LOAD DATA
    # ==========================================================

    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_placement_data.csv"
    )

    df = load_and_validate_data(DATA_PATH)

    # Remove rows with missing values
    df_clean = df.dropna(
        subset=[
            "cgpa",
            "communication_skills",
            "salary_package_lpa"
        ]
    ).copy()

    # ==========================================================
    # 2. DEFINE INPUT AND OUTPUT
    # ==========================================================

    # L = 2 input features
    feature_cols = [
        "cgpa",
        "communication_skills"
    ]

    # M = 1 output
    target_col = "salary_package_lpa"

    X_raw = df_clean[feature_cols].values

    y = df_clean[target_col].values.reshape(-1, 1)

    N = X_raw.shape[0]

    print(
        f"Loaded {N} data points with "
        f"input dimension L = {X_raw.shape[1]} "
        f"and output dimension M = {y.shape[1]}"
    )

    # ==========================================================
    # 3. CREATE DESIGN MATRIX
    # ==========================================================

    # Add a column of ones for the intercept/bias
    #
    # X_design = [1, CGPA, Communication Skills]

    X_design = np.hstack(
        [
            np.ones((N, 1)),
            X_raw
        ]
    )

    # ==========================================================
    # 4. STANDARD LEAST SQUARES
    # ==========================================================

    # Normal Equation:
    #
    # w = (X^T X)^-1 X^T y

    XT_X = np.dot(
        X_design.T,
        X_design
    )

    # Check whether matrix is invertible
    try:

        XT_X_inv = np.linalg.inv(
            XT_X
        )

    except np.linalg.LinAlgError:

        print(
            "Matrix is singular. "
            "Using pseudo-inverse."
        )

        XT_X_inv = np.linalg.pinv(
            XT_X
        )

    # Calculate X^T y

    XT_y = np.dot(
        X_design.T,
        y
    )

    # Calculate optimal weights

    w_optimal = np.dot(
        XT_X_inv,
        XT_y
    )

    # ==========================================================
    # 5. DISPLAY MODEL PARAMETERS
    # ==========================================================

    print(
        "\n--- Optimal Model Parameters "
        "(Weights & Bias) ---"
    )

    print(
        f"Intercept (w0): "
        f"{w_optimal[0, 0]:.4f}"
    )

    print(
        f"Coefficient for {feature_cols[0]} "
        f"(w1): {w_optimal[1, 0]:.4f}"
    )

    print(
        f"Coefficient for {feature_cols[1]} "
        f"(w2): {w_optimal[2, 0]:.4f}"
    )

    # ==========================================================
    # 6. CALCULATE PREDICTIONS
    # ==========================================================

    y_pred = np.dot(
        X_design,
        w_optimal
    )

    # ==========================================================
    # 7. CALCULATE ERROR
    # ==========================================================

    # E_w = 0.5 * sum((y_pred - y)^2)

    E_w = 0.5 * np.sum(
        (y_pred - y) ** 2
    )

    print(
        f"Minimized Error (E_w): "
        f"{E_w:.4f}"
    )

    # ==========================================================
    # 8. CREATE 3D VISUALIZATION
    # ==========================================================

    os.makedirs(
        "reports/figures",
        exist_ok=True
    )

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # ----------------------------------------------------------
    # Actual data points
    # ----------------------------------------------------------

    # y.ravel() converts (100000, 1) into (100000,)
    # This prevents the memory error during plotting.

    ax.scatter(
        X_raw[:, 0],
        X_raw[:, 1],
        y.ravel(),
        alpha=0.6,
        label="Actual Data Points"
    )

    # ----------------------------------------------------------
    # Create meshgrid for regression plane
    # ----------------------------------------------------------

    x1_surf = np.linspace(
        X_raw[:, 0].min(),
        X_raw[:, 0].max(),
        20
    )

    x2_surf = np.linspace(
        X_raw[:, 1].min(),
        X_raw[:, 1].max(),
        20
    )

    x1_mesh, x2_mesh = np.meshgrid(
        x1_surf,
        x2_surf
    )

    # ----------------------------------------------------------
    # Calculate values for regression plane
    # ----------------------------------------------------------

    # y = w0 + w1*x1 + w2*x2

    y_mesh = (
        w_optimal[0, 0]
        + w_optimal[1, 0] * x1_mesh
        + w_optimal[2, 0] * x2_mesh
    )

    # ----------------------------------------------------------
    # Plot regression plane
    # ----------------------------------------------------------

    ax.plot_surface(
        x1_mesh,
        x2_mesh,
        y_mesh,
        alpha=0.3,
        edgecolor="none"
    )

    # ==========================================================
    # 9. LABEL THE GRAPH
    # ==========================================================

    ax.set_xlabel(
        "CGPA (Feature 1)"
    )

    ax.set_ylabel(
        "Communication Skills (Feature 2)"
    )

    ax.set_zlabel(
        "Salary Package LPA (Target)"
    )

    ax.set_title(
        "Linear Regression via Standard Least Squares "
        "(P=1, L=2, M=1)"
    )

    plt.tight_layout()

    # ==========================================================
    # 10. SAVE GRAPH
    # ==========================================================

    output_path = (
        "reports/figures/"
        "linear_regression_3d_plane.png"
    )

    plt.savefig(
        output_path
    )

    plt.close()

    print(
        f"\n-> Successfully saved 3D "
        f"regression plot to {output_path}"
    )


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    train_linear_regression_ls()