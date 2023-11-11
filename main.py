import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


class QSTSquareness:
    def __init__(self, location: str, separator: str):
        super().__init__()
        self._data: pd = None
        self._location: str = location
        self._separator: str = separator
        self.features: list[str] = [
            'CLE_QST',
            'MODELE_PILOTAGE',
            # 'DTE_CREA_REC',
            # 'CO_ORIG_COUL',
            # 'AA_COUL',
            # 'NUM_COUL',
            'NUM_BB',
            'TYPE_BB',
            'TYPE_PROF_MONT',
            'DIM_PROF_MONT',
            'TYPE_PROF_LME',
            'DIM_PROF_LME',
            'PCRT_PROF_LME',
            'NUTR_VISE',
            'NUTR_REAL',
            'QST_HISTAR_VISE',
            'CMPGE_LMG',
            'MONTAGE',
            'NUM_COURANT',
            'DBT_AILE_INT_SUP',
            'DBT_AME_SUP',
            'DBT_AILE_INT_SUP_AVANT_EQ',
            'DBT_AME_SUP_AVANT_EQ',
            'DBT_AILE_INT_SUP_REAL',
            'DBT_AME_SUP_REAL',
            'MODE_ENF_DP',
            'CORR_EQ_AILE_SUP_REAL',
            'CORR_EQ_AME_SUP_REAL',
            'CLE_STAT',
            'NBRE_DEF16_BM',
            'NBRE_BF'
        ]
        self.targets: list[str] = [
            'MOD_DBT_AILE_SUP',
            'MOD_DBT_AME_SUP'
        ]
        self.categorical_cols: list[str] = [
            'MODELE_PILOTAGE',
            # 'DTE_CREA_REC',
            'TYPE_BB',
            'TYPE_PROF_MONT',
            'TYPE_PROF_LME',
            'CMPGE_LMG',
            'MONTAGE',
            'MODE_ENF_DP'
        ]
        self.na_cols: list[str] = [
            'MOD_DBT_AILE_SUP',
            'MOD_DBT_AME_SUP'
        ]
        self.x: pd = None
        self.y: pd = None

    def load_data(self) -> None:
        """ load data in a panda dataframe and follows to drop NA values and encoder categorical values"""
        self._data = pd.read_csv(self._location, sep=self._separator)
        self._drop_nas()
        self._encode_labels()
        self.x = self._data[self.features]
        self.y = self._data[self.targets]

    def _drop_nas(self) -> None:
        """ drop na values"""
        self._data = self._data.dropna(subset=self.na_cols)

    def _encode_labels(self) -> None:
        """ Encode categorical columns """
        label_encoders = {}
        for col in self.categorical_cols:
            label_encoders[col] = LabelEncoder()
            self._data[col] = label_encoders[col].fit_transform(self._data[col].astype(str))

    def split_train_test(self, test_size: float, random_state: int) -> tuple:
        """ returns X_train, X_test, Y_train, Y_test """
        return train_test_split(self.x, self.y,
                                test_size=test_size,
                                random_state=random_state)

    @staticmethod
    def model_xgb(objective: str, alpha: int, tree_method: str, eval_metric: str, subsample: float) -> xgb:

        model = xgb.XGBRegressor(
            objective=objective,
            alpha=alpha,
            tree_method=tree_method,
            eval_metric=eval_metric,
            subsample=subsample
        )
        return model


def main() -> None:
    estimator = QSTSquareness('data\\Collecte.csv', ',')
    estimator.load_data()

    X_train, X_test, y_train, y_test = estimator.split_train_test(test_size=0.3, random_state=42)

    model = estimator.model_xgb('reg:squarederror',
                                9,
                                'exact',
                                'rmse',
                                0.9
                                )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    # results = model.evals_result()  # or verbose

    predictions = model.predict(X_test)


if __name__ == '__main__':
    main()
