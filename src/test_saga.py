import unittest
import constants
import test_saga_data as td
import saga_io as io
import saga_operations as so
import numpy as np


def give_me_means(input_df: dict, keys_list: list, means_list: list):
    return all(
        [
            input_df[keys_list[index]].mean() == means_list[index]
            for index in range(len(means_list))
        ]
    )


class TestSaga(unittest.TestCase):
    def setUp(self):
        self.df = io.load_csv(
            path=constants.DF_PATH,
            usecols=constants.COLUMNS_IN_USE,
        )
        self.key_df = io.load_csv(
            path=constants.KEY_PATH,
            usecols=["Timestamp", "Value.Seconds", "Value.Value"],
        )

        self.df_dict = so.deinterleave(self.df, start=constants.START_DEINTERLEAVE)
        self.regions_dict = so.give_me_relevant_regions(self.df_dict)
        self.smooth_dict = so.smooth_signals(self.regions_dict)
        self.corrected_dict = so.correct_signals(self.smooth_dict)
        self.pellets = so.give_me_pellets(self.df_dict["reference"], self.key_df)
        self.fit_dict = so.fit(self.df_dict, self.corrected_dict)
        self.fitsub_dict = so.subtract_fits(
            self.df_dict, self.fit_dict, self.corrected_dict
        )
        self.chopped_dict = so.chop_up(self.fitsub_dict, self.pellets)
        self.avg_dff_dict = so.avg_dff(self.chopped_dict)
        self.all_trials_dict, self.error_dict = so.normalized_signal(self.chopped_dict, self.avg_dff_dict)

    def test_load_csv(self):
        self.assertEqual(self.df.shape[0], 304723)
        self.assertEqual(self.key_df.shape[0], 56)

    def test_deinterleave(self):
        self.assertEqual(len(self.df_dict), 3)
        self.assertTrue(
            self.df_dict["green"].loc[0, "LedState"] == 2
            and self.df_dict["reference"].loc[0, "LedState"] == 1
            and self.df_dict["red"].loc[0, "LedState"] == 4,
        )
        self.assertTrue(
            len(self.df_dict["green"])
            and len(self.df_dict["red"])
            and len(self.df_dict["reference"]) == 94773,
        )

    def test_relevant_regions(self):
        self.assertTrue(
            give_me_means(
                self.regions_dict, td.KEYS_LIST, td.MEANS_LIST_RELEVANT_REGIONS
            )
        )

    def test_smooth_signals(self):
        self.assertTrue(
            give_me_means(self.smooth_dict, td.KEYS_LIST, td.MEANS_LIST_SMOOTH_SIGNALS)
        )

    def test_correct_signals(self):
        self.assertTrue(
            give_me_means(
                self.corrected_dict, td.KEYS_LIST_ABBREV, td.MEANS_LIST_CORRECT_SIGNALS
            )
        )

    def test_pellets(self):
        self.assertTrue(np.array_equiv(self.pellets, td.PELLETS_LIST))

    def test_fit(self):
        self.assertTrue(
            give_me_means(self.fit_dict, td.KEYS_LIST_ABBREV, td.MEANS_LIST_FIT)
        )

    def test_fitsub(self):
        self.assertTrue(
            give_me_means(self.fitsub_dict, td.KEYS_LIST_ABBREV, td.MEANS_LIST_FITSUB)
        )

    def test_chop_up(self):
        sum = 0
        for value in self.chopped_dict.values():
            sum += value.mean()
        self.assertEqual(sum, 112.22747012012492)

    def test_avgdff(self):
        sum = 0
        for value in self.avg_dff_dict.values():
            sum += value
        self.assertEqual(sum, 112.79376994493873)

    def test_normalized(self):
        self.assertEqual(self.all_trials_dict['green_right'].to_numpy().sum(), 12760.843552428654)
        self.assertEqual(self.all_trials_dict['green_left'].to_numpy().sum(), 2218.3169049663034)
        # self.assertEqual(self.all_trials_dict['red_right'].to_numpy().sum(), -3851.6706572333715)
        self.assertEqual(self.all_trials_dict['red_left'].to_numpy().sum(), 7070.43493498786)
        total = 0
        for value in self.error_dict.values():
            total += value.sum()
        self.assertEqual(total, 1178.0017785029233)


if __name__ == "__main__":
    unittest.main()
