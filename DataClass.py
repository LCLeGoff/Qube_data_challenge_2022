import pandas as pd
import numpy as np


class XYClass:
    def __init__(self, x_df: pd.DataFrame, y_df: pd.DataFrame):
        """
        CLass providing x and y as DataFrames or arrays. x and y are indexed by dates and stock ids
        :param x_df:
        :param y_df:
        """

        assert x_df.index.equals(y_df.index), 'x and y not same index'
        assert x_df.index.names == ['date', 'stocksID'], 'x index name not correct :' + str(x_df.index.name)

        self.x_df = x_df
        self.y_df = y_df

        self.x_tab = self.x_df.to_numpy()
        self.y_tab = self.y_df.to_numpy()


class DataClass:
    def __init__(
            self, root: str = 'data/', prop_train=0.5, nb_train_set=1, split_method: str = None):
        """
        This class get x and y from the 'X_train.csv' and 'Y_train.csv' files.
        It can also provide train and test dataset by splitting x and y.
        The split can be:
            - random split: the data (X, Y) is shuffled and then split in two.
            - "date-wise": the dates are shuffled and split in two.
            - "stock-wise": the stocks are shuffled and split in two.
        All dataset are provided as XYClasses
        :param root: address for the repertory containing the files 'X_train.csv' and 'Y_train.csv'
        :param prop_train: proportion of the stocks, which will constitute the train dataset.
         The test dataset will contain 1-prop_train of the stocks
        :param nb_train_set: the training dataset can be divided in nb_train_set parts
        :param split_method: Method used to split de data into train and test dataset. If None, split is made
        """

        self.root = root
        self.prop_train = prop_train
        self.nb_train_set = nb_train_set
        self.split_method = split_method

        x_df, y_df = self._get_data()
        self.xy = XYClass(x_df=x_df, y_df=y_df)

        self.stocks = self.xy.x_df.index.get_level_values('stocksID').sort_values().unique()
        self.dates = self.xy.x_df.index.get_level_values('date').sort_values().unique()

        self.xy_train = None
        self.xy_test = None
        if split_method is not None:
            self._get_train_test_data()

    def _get_data(self) -> [pd.DataFrame, pd.DataFrame]:
        """
        get X and Y from files 'X_train.csv' and 'Y_train.csv'
        :return: X and Y as DataFrames
        """

        x_df = pd.read_csv(self.root + 'X_train.csv', index_col=0, sep=',')
        x_df.columns = x_df.columns.astype(int)
        x_df.columns.name = 'date'
        x_df = pd.concat(objs=[x_df.T.shift(i + 1).stack(dropna=False) for i in range(250)], axis=1).dropna()
        x_df.columns = pd.Index(range(1, 251), name='timeLag')

        y_df = pd.read_csv(self.root + 'Y_train.csv', index_col=0, sep=',')
        y_df.columns = y_df.columns.astype(int)
        y_df.columns.name = 'date'
        y_df = y_df.unstack()

        return x_df, y_df

    def _get_train_test_data(self):
        """
        split the dataset in train and test dataset
        """

        idx = self._get_idx()
        np.random.shuffle(idx)
        train_lg = int(len(idx) * self.prop_train)
        idx_test = idx[train_lg:]

        x, y = self._get_xy_sub_dataframe_from_idx(idx_test)
        self.xy_test = XYClass(x, y)

        idx_train = idx[:train_lg]
        self._get_train_data(idx_train)

    def _get_train_data(self, idx: list):
        """
        Provide the train dataset from the index idx. The train set is divided in parts, if self.nb_train_set != 1
        :param idx: idx of the train dataset
        """
        if self.nb_train_set == 1:
            x, y = self._get_xy_sub_dataframe_from_idx(idx)
            self.xy_train = XYClass(x, y)
        else:

            self.xy_train = []

            train_lg = len(idx) // self.nb_train_set
            for i in range(self.nb_train_set):
                idx_train = idx[train_lg * i:train_lg * (i + 1)]
                x, y = self._get_xy_sub_dataframe_from_idx(idx_train)
                self.xy_train.append(XYClass(x, y))

    def _get_idx(self) -> list:
        """
        Get the indexes used to split the dataset.
        :return: list of the indexes
        """
        if self.split_method == 'stock':
            return self._get_stock_idx()
        elif self.split_method == 'random':
            return self._get_iloc_idx()
        elif self.split_method == 'date':
            return self._get_date_idx()
        else:
            raise NameError(self.split_method + ' not known as train_method')

    def _get_xy_sub_dataframe_from_idx(self, idx: list) -> [pd.DataFrame, pd.DataFrame]:
        """
        Give the sub dataFrame of X and Y from indexes idx for iloc, stock or date index type
        :param idx: list of the indexes
        :return: tuple of the X and Y DataFrame
        """
        if self.split_method == 'stock':
            return self._get_xy_sub_dataframe_from_stock_idx(idx)
        elif self.split_method == 'random':
            return self._get_xy_sub_dataframe_from_iloc_idx(idx)
        elif self.split_method == 'date':
            return self._get_xy_sub_dataframe_from_date_idx(idx)
        else:
            raise self.split_method + ' not known as train_method'

    def _get_xy_sub_dataframe_from_iloc_idx(self, idx: list) -> [pd.DataFrame, pd.DataFrame]:
        """
        Give the sub dataFrame of X and Y from indexes idx for iloc index type
        :param idx: list of the indexes
        :return: tuple of the X and Y DataFrame
        """
        x = self.xy.x_df.iloc[idx]
        y = self.xy.y_df.iloc[idx]
        return x, y

    def _get_xy_sub_dataframe_from_stock_idx(self, idx: list) -> [pd.DataFrame, pd.DataFrame]:
        """
        Give the sub dataFrame of X and Y from indexes idx for stock index type
        :param idx: list of the indexes
        :return: tuple of the X and Y DataFrame
        """
        x = self.xy.x_df.loc[pd.IndexSlice[:, idx], :]
        y = self.xy.y_df.loc[pd.IndexSlice[:, idx]]
        return x, y

    def _get_xy_sub_dataframe_from_date_idx(self, idx: list) -> [pd.DataFrame, pd.DataFrame]:
        """
        Give the sub dataFrame of X and Y from indexes idx for date index type
        :param idx: list of the indexes
        :return: tuple of the X and Y DataFrame
        """
        x = self.xy.x_df.loc[idx]
        y = self.xy.y_df.loc[idx]
        return x, y

    def _get_iloc_idx(self) -> list:
        """
        Get the indexes used to iloc split the dataset.
        :return: list of the indexes
        """
        return list(range(len(self.xy.x_df)))

    def _get_stock_idx(self) -> list:
        """
        Get the indexes used to stock the dataset.
        :return: list of the indexes
        """
        return list(self.stocks.copy())

    def _get_date_idx(self) -> list:
        """
        Get the indexes used to date the dataset.
        :return: list of the indexes
        """
        return list(self.dates.copy())
