"""
Features anonymisation script

This code anonymise your features by changing column names and applying transformation.
The only way to de-anonymise the features is by using the de-anonymise.pkl file.

Requirements:
    pandas, scikit-learn

Output:
    a CSV file named "QuantModelReport.csv" with anonymised features
    a pkl file named "de-anonymise.pkl" which contains all necessary data for de-anonymisation

__author__ = "David Forino"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "David Forino"
__email__ = "d.forino@quanted.com"
__status__ = "Production"
"""

import pandas as pd
from sklearn.preprocessing import PowerTransformer
from pandas.api.types import is_numeric_dtype
import pickle


def main(df:pd.DataFrame) -> None:
    df_copy = df.copy()
    exclude_cols = ["PredictionDT", "PredictedValue", "RealValue"]

    # remove non numerical features, they can't be processed as of now
    for column in df.columns:
        if column in exclude_cols:
            continue
        if not is_numeric_dtype(df_copy[column]):
            df_copy.drop(columns=[column], inplace=True)

    # anonymise feature names
    columns = df_copy.columns.to_list()
    new_columns = [f"Feature_{i}" for i, col in enumerate(columns) if col not in exclude_cols]
    anonymised_cols = [f"Feature_{i}" if col not in exclude_cols else col for i, col in enumerate(columns)]
    df_copy.columns = anonymised_cols

    # anonymise features data
    transformer = PowerTransformer(method='yeo-johnson', standardize=True, copy=True)
    df_copy.loc[:, new_columns] = transformer.fit_transform(df_copy[new_columns])

    # store anonymised CSV file
    df_copy.to_csv("QuantModelReport.csv", index=False)

    # store old feature names and transformer to reverse the anonymisation process
    with open("de-anonymise.pkl", "wb") as file:
        pickle.dump(
            obj={"Features_old":columns, "Features_new":anonymised_cols, "Transformer":transformer}, 
            file=file, protocol=pickle.HIGHEST_PROTOCOL
            )


if __name__ == "__main__":
    my_df = pd.read_csv("MyCSVFile.csv")
    main(df=my_df)
