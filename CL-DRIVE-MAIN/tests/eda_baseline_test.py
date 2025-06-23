import unittest

import numpy as np
import pandas as pd

from utils.data import read_data_with_structure


class TestData(unittest.TestCase):
    def test_data(self):
        dataset = "./cl_drive/"
        data = read_data_with_structure(
            dataset,
            drop_na=True,
            load_baseline=True,
            participant="1241",
            modality="EDA",
        )

        predefined_row = [
            np.float64(94.8087685),
            np.nan,
            np.nan,
            np.nan,
            np.float64(34671.0),
            np.float64(559.1970503891849),
            np.float64(1.7882783882783877),
            "1241",
            "EDA",
            1,
        ]

        columns = [
            "Timestamp",
            "GSR RAW",
            "GSR Resistance CAL",
            "GSR Conductance CAL",
            "GSR RAW.1",
            "GSR Resistance CAL.1",
            "GSR Conductance CAL.1",
            "Participant_ID",
            "Modality",
            "Complexity_Level",
        ]

        row_to_test = data[
            (data["Timestamp"] == predefined_row[0])
            & (data["Complexity_Level"] == predefined_row[9])
        ].iloc[0]
        row_to_test_list = row_to_test[columns].tolist()

        for idx, (actual, expected) in enumerate(zip(row_to_test_list, predefined_row)):
            with self.subTest(column=columns[idx]):
                if pd.isna(expected):
                    self.assertTrue(
                        pd.isna(actual), f"Mismatch in column {columns[idx]}"
                    )
                else:
                    self.assertEqual(
                        actual, expected, f"Mismatch in column {columns[idx]}"
                    )


if __name__ == "__main__":
    unittest.main()
