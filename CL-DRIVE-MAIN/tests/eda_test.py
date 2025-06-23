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
            participant="1717",
            modality="EDA",
        )

        predefined_row = [
            np.float64(222.5760566),
            np.float64(1197.0),
            np.float64(53.3265306122449),
            np.float64(18.752391886720247),
            np.nan,
            np.nan,
            np.nan,
            "1717",
            "EDA",
            6,
            3,
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
