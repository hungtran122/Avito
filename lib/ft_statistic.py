from tqdm import tqdm


class FeaturesStatistics():
    def __init__(self, cols):
        self._stats = None
        self._agg_cols = cols

    def fit(self, df):
        '''
        Compute the mean and std of some features from a given data frame
        '''
        self._stats = {}

        # For each feature to be aggregated
        for c in tqdm(self._agg_cols, total=len(self._agg_cols)):
            # Compute the mean and std of the deal prob and the price.
            gp = df.groupby(c)[['deal_probability', 'price']]
            desc = gp.describe()
            self._stats[c] = desc[[('deal_probability', 'mean'), ('deal_probability', 'std'),
                                   ('price', 'mean'), ('price', 'std')]]

    def transform(self, df):
        '''
        Add the mean features statistics computed from another dataset.
        '''
        # For each feature to be aggregated
        for c in tqdm(self._agg_cols, total=len(self._agg_cols)):
            # Add the deal proba and price statistics corrresponding to the feature
            df[c + '_dp_mean'] = df[c].map(self._stats[c][('deal_probability', 'mean')])
            df[c + '_dp_std'] = df[c].map(self._stats[c][('deal_probability', 'std')])
            df[c + '_price_mean'] = df[c].map(self._stats[c][('price', 'mean')])
            df[c + '_price_std'] = df[c].map(self._stats[c][('price', 'std')])

            df[c + '_to_price'] = df.price / df[c + '_price_mean']
            df[c + '_to_price'] = df[c + '_to_price'].fillna(1.0)

    def fit_transform(self, df):
        '''
        First learn the feature statistics, then add them to the dataframe.
        '''
        self.fit(df)
        self.transform(df)