class CONFIG:
    target = "responder_6"
    main = "C:/Users/edmun/OneDrive/Desktop/Personal-Projects/Kaggle/Jane Street Real Time Market Data Forecasting/"

    # Number of dates to skip from the beginning of the dataset
    skip_dates = 500

    # Define the feature names based on the number of features (79 in this case)
    feature_names = [f"feature_{i:02d}" for i in range(79)]
    categorical_feature = [f"feature_{i:02d}" for i in range(9,12)]
    numerical_feature = [f"feature_{i:02d}" for i in range(79) if i not in range(9,12)]
    exogeneous_features = [
        "sin_time_id",
        "cos_time_id",
        "sin_time_id_halfday",
        "cos_time_id_halfday",
    ]
    lag_features = [f"responder_{idx}_lag_1" for idx in range(9)]

    N_fold = 5
