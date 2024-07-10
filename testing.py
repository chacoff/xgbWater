from training import SquarenessEstimator

estimator = SquarenessEstimator('dataS\\dev.csv',
                                ',',
                                'modelsS\\model_Yr',
                                False)
estimator.load_data()
model = estimator.load_model()

input_dev: int = estimator.x.shape[0]
for i in range(0, input_dev):
    single_input = estimator.x.iloc[[i]]
    print(f'INPUT DATA:\n{single_input.transpose()}')
    predictions = model.predict(single_input)

    print(predictions)

    # print(f'\nMOD_DBT_AILE_SUP = {round(float(predictions[0][0]), 4)}\n'
    #       f'MOD_DBT_AME_SUP = {round(float(predictions[0][1]), 4)}')