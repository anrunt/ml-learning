from pandas import DataFrame, Series
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text


def train_regression(X_train: DataFrame, y_train: Series) -> LinearRegression:
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    return reg


def train_baseline(X_train: DataFrame, y_train: Series) -> DummyRegressor:
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    return baseline


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
    }


def main() -> None:
    # Load data
    X: DataFrame
    y: Series
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    print(X.assign(target=y).head())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )



    reg = train_regression(X_train, y_train)
    baseline = train_baseline(X_train, y_train)

    # Predict using the fitted models without training them again.
    y_pred_train = reg.predict(X_train)
    y_pred = reg.predict(X_test)

    y_pred_baseline_train = baseline.predict(X_train)
    y_pred_baseline = baseline.predict(X_test)

    regression_metrics = DataFrame({
        "Trening": calculate_metrics(y_train, y_pred_train),
        "Test": calculate_metrics(y_test, y_pred),
    })
    baseline_metrics = DataFrame({
        "Trening": calculate_metrics(y_train, y_pred_baseline_train),
        "Test": calculate_metrics(y_test, y_pred_baseline),
    })

    print("\nLinear Regression:")
    print(regression_metrics.round(4).to_string())

    print("\nBaseline (średnia z y_train):")
    print(baseline_metrics.round(4).to_string())

    results = DataFrame({
        "Actual": y_test,
        "Predicted": y_pred,
        "Baseline": y_pred_baseline,
    })
    results["Error"] = results["Actual"] - results["Predicted"]
    results["Baseline error"] = results["Actual"] - results["Baseline"]

    print("\nPorównanie dla pierwszych 10 pacjentów:")
    print(results.head(10).round(2))

    # cross_val
    cross_val_scores = cross_val_score(LinearRegression(), X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
    mae_folds = -cross_val_scores

    print("\n")
    print("MAE for every fold: ", mae_folds)
    print("MAE mean: ", mae_folds.mean())
    print("MAE std: ", mae_folds.std())

    # ridge
#    alphas = [0.001, 0.01, 0.1, 1, 10, 100]
#    for alpha in alphas:
#        score = cross_val_score(Ridge(alpha=alpha), X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
#        mae_folds_r = -score
#
#        print("\n")
#        print("Ridge for alpha: ", alpha)
#        print("MAE for every fold: ", mae_folds_r)
#        print("MAE mean: ", mae_folds_r.mean())
#        print("MAE std: ", mae_folds_r.std())


    ridge = Ridge(alpha=0.01)
    ridge.fit(X_train, y_train)

    coefficients = DataFrame({
        "Feature": X_train.columns,
        "LinearRegression": reg.coef_,
        "Ridge": ridge.coef_,
    })

    print(coefficients.round(2).to_string(index=False))

    print("Linear Regression:", np.linalg.norm(reg.coef_))
    print("Ridge:", np.linalg.norm(ridge.coef_))

    y_pred_ridge = ridge.predict(X_test)

    print(calculate_metrics(y_test, y_pred_ridge))


    tree = DecisionTreeRegressor(max_depth=2, random_state=42)

    tree.fit(X_test, y_test)

    print(
        export_text(
            tree,
            feature_names=list(X_train.columns)
        )
    )

# Correlation 
    print(X_train.corr().round(2))
if __name__ == "__main__":
    main()
