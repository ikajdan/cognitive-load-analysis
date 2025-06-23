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
            participant="1417",
            modality="ECG",
        )

        predefined_row = [
            np.float64(298.80054035),
            np.float64(-27456.0),
            np.float64(-1.980171439668112),
            np.float64(-97261.0),
            np.float64(-7.014621736362188),
            np.float64(54069.0),
            np.float64(3.899544346278232),
            "1417",
            "ECG",
            1,
            3,
        ]

        columns = [
            "Timestamp",
            "ECG LL-RA RAW",
            "ECG LL-RA CAL",
            "ECG LA-RA RAW",
            "ECG LA-RA CAL",
            "ECG Vx-RL RAW",
            "ECG Vx-RL CAL",
            "Participant_ID",
            "Modality",
            "Complexity_Level",
            "Selfassessed_Level",
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
