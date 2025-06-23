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
            participant="1105",
            modality="Gaze",
        )

        predefined_row = [
            np.float64(228.724),
            np.float64(5.800000190734863),
            np.float64(4.550000190734863),
            np.float64(858.0),
            np.float64(666.0),
            np.float64(858.0),
            np.float64(666.0),
            np.float64(65.5999984741211),
            np.float64(-54.06999969482422),
            np.float64(652.8200073242188),
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0676999986171722),
            np.float64(-0.0274000000208616),
            np.float64(0.9973000288009644),
            np.float64(0.123599998652935),
            np.float64(-0.0723000019788742),
            np.float64(0.9897000193595886),
            np.float64(658.3319091796875),
            np.float64(0.0),
            np.float64(858.0),
            np.float64(666.0),
            np.float64(855.0),
            np.float64(631.0),
            np.float64(1000.0),
            np.float64(84.7994534728498),
            np.float64(445.134698138175),
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.float64(948.0),
            np.float64(948.0),
            np.float64(228694.0),
            np.float64(228734.0),
            np.float64(40.0),
            np.float64(2.44682122297677),
            np.float64(84.7994534728498),
            np.float64(2374.66618383916),
            np.nan,
            np.float64(81.4692343900519),
            "1105",
            "Gaze",
            9,
            9,
        ]

        columns = [
            "Timestamp",
            "ET_PupilLeft",
            "ET_PupilRight",
            "ET_GazeLeftx",
            "ET_GazeLefty",
            "ET_GazeRightx",
            "ET_GazeRighty",
            "ET_Gaze3DX",
            "ET_Gaze3DY",
            "ET_Gaze3DZ",
            "ET_ValidityLeftEye",
            "ET_ValidityRightEye",
            "ET_GazeDirectionLeftX",
            "ET_GazeDirectionLeftY",
            "ET_GazeDirectionLeftZ",
            "ET_GazeDirectionRightX",
            "ET_GazeDirectionRightY",
            "ET_GazeDirectionRightZ",
            "ET_Distance3D",
            "Blink detected (binary)",
            "Gaze X",
            "Gaze Y",
            "Interpolated Gaze X",
            "Interpolated Gaze Y",
            "Interpolated Distance",
            "Gaze Velocity",
            "Gaze Acceleration",
            "Fixation Index",
            "Fixation Index by Stimulus",
            "Fixation X",
            "Fixation Y",
            "Fixation Start",
            "Fixation End",
            "Fixation Duration",
            "Fixation Dispersion",
            "Saccade Index",
            "Saccade Index by Stimulus",
            "Saccade Start",
            "Saccade End",
            "Saccade Duration",
            "Saccade Amplitude",
            "Saccade Peak Velocity",
            "Saccade Peak Acceleration",
            "Saccade Peak Deceleration",
            "Saccade Direction",
            "Participant_ID",
            "Modality",
            "Complexity_Level",
            "Selfassessed_Level",
        ]

        row_to_test = data[
            (data["Timestamp"] == predefined_row[0])
            & (data["Complexity_Level"] == predefined_row[-2])
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
