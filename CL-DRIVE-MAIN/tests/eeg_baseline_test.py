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
            participant="1892",
            modality="EEG",
        )

        predefined_row = [
            np.float64(94.42578125),
            np.float64(25.87890625),
            np.float64(-27.34375),
            np.float64(-44.921875),
            np.float64(36.1328125),
            "1892",
            "EEG",
            5,
        ]

        columns = [
            "Timestamp",
            "TP9",
            "AF7",
            "AF8",
            "TP10",
            "Participant_ID",
            "Modality",
            "Complexity_Level",
        ]

        row_to_test = data[
            (data["Timestamp"] == predefined_row[0])
            & (data["Complexity_Level"] == predefined_row[7])
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
