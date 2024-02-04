class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob

    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            indices = np.random.choice(data_length, size=data_length, replace=True)
            self.indices_list.append(indices)

    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.

        example:

        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(
            data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            data_bag, target_bag = data[self.indices_list[bag]], target[self.indices_list[bag]]  # Your Code Here
            self.models_list.append(model.fit(data_bag, target_bag))  # store fitted models here
        if self.oob:
            self.data = data
            self.target = target

    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        predictions = []
        for model in self.models_list:
            prediction = model.predict(data)
            predictions.append(prediction)

        return np.mean(predictions, axis=0)

    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]

        for i, model in enumerate(self.models_list):
            # Get the indices that were used for training this model
            oob_indices = [idx for idx in range(len(self.data)) if idx not in self.indices_list[i]]

            # Make predictions for the OOB indices
            predictions = model.predict([self.data[idx] for idx in oob_indices])

            # Store the predictions for each OOB index
            for j, idx in enumerate(oob_indices):
                list_of_predictions_lists[idx].append(predictions[j])

        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)

    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        averaged_predictions = []
        for predictions in self.list_of_predictions_lists:
            if len(predictions) > 0:
                averaged_predictions.append(np.mean(predictions))
            else:
                averaged_predictions.append(None)
        self.averaged_predictions = averaged_predictions

    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        squared_errors = []
        for i, prediction in enumerate(self.averaged_predictions):
            if prediction is not None:
                squared_errors.append((self.target[i] - prediction) ** 2)
        if len(squared_errors) > 0:
            return np.mean(squared_errors)
        else:
            return 0.0