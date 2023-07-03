# pyright: basic

import json
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import polars as pl
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from aliases import (
    IS_CO_COMP,
    PROBA_CO_COMP,
    PROBA_NON_CO_COMP,
    PROTEIN_U,
    PROTEIN_V,
    WEIGHT,
    XVAL_ITER,
)
from assertions import assert_no_zero_weight


class SupervisedWeighting:
    """
    Supervised co-complex probability weighting method.
    """

    def __init__(self, model: XGBClassifier, name: str, dip: bool):
        self.label = IS_CO_COMP
        self.model = model
        self.name = name
        self.prefix = "dip_" if dip else ""

    def tune(
        self,
        params_grid: Dict[str, Union[List[str], List[int], List[float]]],
        X_train: np.ndarray[Any, Any],
        y_train: np.ndarray[Any, Any],
    ) -> Dict[str, Union[int, float]]:
        """
        Tune the model via grid searching.

        Args:
            params_grid (Dict[str, Union[List[str], List[int], List[float]]]): Parameter grid.
            X_train (np.ndarray[Any, Any]): Training dataset X values.
            y_train (np.ndarray[Any, Any]): Training dataset y values.
        """
        clf = GridSearchCV(
            self.model,
            params_grid,
            n_jobs=-1,
            cv=5,
            scoring="neg_log_loss",
            refit=True,
        )
        clf.fit(X_train, y_train)
        print(f"Best parameters: {clf.best_params_}")
        print(f"Best score: {clf.best_score_}")
        self.model = clf.best_estimator_
        return clf.best_params_

    def weight(
        self,
        df_composite: pl.DataFrame,
        df_train_labeled: pl.DataFrame,
        xval_iter: int,
        tune: bool,
        params_grid: Dict[str, Union[List[str], List[int], List[float]]],
        use_pretuned_params: bool,
    ) -> Tuple[pl.DataFrame, Dict[str, Union[float, int]]]:
        """
        Weight the composite network based on co-complex probability.

        Args:
            df_composite (pl.DataFrame): Composite protein network.
            df_train_labeled (pl.DataFrame): Labeled dataset via the training set.
            xval_iter (int): Cross-val iteration.
            tune (bool): Whether to tune the model or not.
            params_grid (Dict[str, Union[List[str], List[int], List[float]]]): Parameter grid for tuning.
            use_pretuned_params (bool): Whether to use a saved set of hyperparameters.

        Returns:
            Tuple[pl.DataFrame, Dict[str, Union[float, int]]]: Weighted network and feature importances
        """
        selected_features = df_composite.select(
            pl.exclude([PROTEIN_U, PROTEIN_V, self.label])
        ).columns

        print(f"Weighting model: {self.name}")
        print(f"Selected features: {selected_features}")

        df_feat_label = df_train_labeled.join(
            df_composite, on=[PROTEIN_U, PROTEIN_V], how="left"
        )
        X_train = df_feat_label.select(selected_features).to_numpy()
        y_train = df_feat_label.select(self.label).to_numpy().ravel()

        n_samples = X_train.shape[0]
        co_comp_samples = y_train[y_train == 1].shape[0]
        non_co_comp_samples = n_samples - co_comp_samples

        print(
            f"Train samples: {n_samples} | Co-comp: {co_comp_samples} | Non-co-comp: {non_co_comp_samples}"
        )

        if tune:
            print("Performing hyperparameter tuning...")
            params_grid["random_state"] = [xval_iter]  # for reproducibility
            best_params = self.tune(params_grid, X_train, y_train)
            print("Hyperparameter tuning done!")
            params_file = (
                f"../data/training/{self.prefix}xgw_params_iter{xval_iter}.json"
            )
            print(f"Writing best parameters to {params_file}")
            with open(params_file, "w") as file:
                json.dump(best_params, file, indent=4)
            print("Training done!")

        elif use_pretuned_params:
            params_file = (
                f"../data/training/{self.prefix}xgw_params_iter{xval_iter}.json"
            )
            with open(params_file) as file:
                print(f"Using pretuned parameters in: {params_file}")
                best_params = json.load(file)
                print(f"Parameters: {best_params}")
                self.model.set_params(**best_params)

            self.model.fit(X_train, y_train)  # training the model
            print("Training done!")

        else:
            # For reproducibility
            self.model.set_params(random_state=xval_iter)

            self.model.fit(X_train, y_train)  # training the model
            print("Training done!")

        params = self.model.get_params()
        print(f"Parameters used for fitting: {params}")

        feature_importances = {
            F: self.model.feature_importances_[idx]
            for idx, F in enumerate(selected_features)
        }

        feature_importances[XVAL_ITER] = xval_iter

        # After learning the parameters, weight all protein pairs
        X = df_composite.select(selected_features).to_numpy()
        ndarr_pred = self.model.predict_proba(X)

        CLASS_PROBA = [PROBA_NON_CO_COMP, PROBA_CO_COMP]

        df_weights = pl.from_numpy(
            ndarr_pred, schema=[CLASS_PROBA[c] for c in self.model.classes_]
        )

        df_w_composite = pl.concat([df_composite, df_weights], how="horizontal").select(
            [PROTEIN_U, PROTEIN_V, PROBA_NON_CO_COMP, PROBA_CO_COMP]
        )

        print("Weighting done!")
        return df_w_composite, feature_importances

    def main(
        self,
        df_composite: pl.DataFrame,
        df_train_labeled: pl.DataFrame,
        xval_iter: int,
        tune: bool,
        params_grid: Dict[str, Union[List[str], List[int], List[float]]],
        use_pretuned_params: bool,
    ) -> Tuple[pl.DataFrame, Dict[str, Union[float, int]]]:
        """
        Main method of SupervisedWeighting.

        Args:
            df_composite (pl.DataFrame): Composite protein network.
            df_train_labeled (pl.DataFrame): Labeled dataset via the training set.
            xval_iter (int): Cross-val iteration.
            tune (bool): Whether to tune the model or not.
            params_grid (Dict[str, Union[List[str], List[int], List[float]]]): Parameter grid for tuning.
            use_pretuned_params (bool): Whether to use a saved set of hyperparameters.

        Returns:
            Tuple[pl.DataFrame, Dict[str, Union[float, int]]]: Weighted network and feature importances
        """
        print()
        df_w_composite, feature_importances = self.weight(
            df_composite,
            df_train_labeled,
            xval_iter,
            tune,
            params_grid,
            use_pretuned_params,
        )

        df_w_composite = (
            df_w_composite.lazy()
            .rename({PROBA_CO_COMP: WEIGHT})
            .select([PROTEIN_U, PROTEIN_V, WEIGHT])
            .collect()
        )

        df_w_composite.write_csv(
            f"../data/weighted/all_edges/cross_val/{self.prefix}{self.name.lower()}_iter{xval_iter}.csv",
            has_header=False,
            separator="\t",
        )

        df_w_20k = df_w_composite.sort(pl.col(WEIGHT), descending=True).head(20_000)
        assert_no_zero_weight(df_w_20k)
        df_w_20k.write_csv(
            f"../data/weighted/20k_edges/cross_val/{self.prefix}{self.name.lower()}_20k_iter{xval_iter}.csv",
            has_header=False,
            separator="\t",
        )
        print()

        return df_w_composite, feature_importances
