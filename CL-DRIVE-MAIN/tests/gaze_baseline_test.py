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
            participant="1744",
            modality="Gaze",
        )

        predefined_row = [
            np.float64(0.079),
            np.float64(4.150000095367432),
            np.float64(4.309999942779541),
            np.float64(1017.0),
            np.float64(796.0),
            np.float64(1017.0),
            np.float64(796.0),
            np.float64(-16.959999084472653),
            np.float64(-80.91999816894531),
            np.float64(415.5),
            np.float64(0.0),
            np.float64(0.0),
            np.float64(-0.0789000019431114),
            np.float64(-0.099600002169609),
            np.float64(0.9919000267982484),
            np.float64(0.0062000001780688),
            np.float64(-0.1700000017881393),
            np.float64(0.9854000210762026),
            np.float64(423.64599609375),
            np.float64(0.0),
            np.float64(1017.0),
            np.float64(796.0),
            np.float64(1017.33333333333),
            np.float64(796.0),
            np.float64(1000.0),
            np.float64(14.6885696795015),
            np.float64(-98.0303467711398),
            np.float64(1.0),
            np.float64(1.0),
            np.float64(1014.02380952381),
            np.float64(786.261904761905),
            np.float64(9.5),
            np.float64(289.0),
            np.float64(279.5),
            np.float64(0.7536242830130669),
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            "1744",
            "Gaze",
            7,
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
        ]

        row_to_test = data[
            (data["Timestamp"] == predefined_row[0])
            & (data["Complexity_Level"] == predefined_row[-1])
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
