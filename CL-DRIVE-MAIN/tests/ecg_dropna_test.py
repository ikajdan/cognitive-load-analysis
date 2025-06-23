import unittest

from utils.data import read_data_with_structure


class TestDropna(unittest.TestCase):
    def test_dropna(self):
        dataset = "./cl_drive/"

        data_no_drop = read_data_with_structure(
            dataset,
            drop_na=False,
            participant="1717",
            modality="ECG",
        )

        data_drop = read_data_with_structure(
            dataset,
            drop_na=True,
            participant="1717",
            modality="ECG",
        )

        rows_no_drop = len(data_no_drop)
        rows_drop = len(data_drop)

        self.assertLessEqual(
            rows_drop,
            rows_no_drop,
            f"Expected fewer rows with drop_na=True, but got {rows_drop} compared to {rows_no_drop}",
        )


if __name__ == "__main__":
    unittest.main()
