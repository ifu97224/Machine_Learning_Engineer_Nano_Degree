class VarSelection:
    """ The variable selection class contains a set of methods to support variable selection in classification / regression modeling """
    
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    
    def __init__(df, target_var, k_features = 10, random_state = 1):
        
        """ Method for initializing a VarSelection object
        
        Args:
        df (Pandas Dataframe)
        target_var (string)
        k_features (int)
        test_size (float)
        random_state (int)
                
        Attributes:
            df (Dataframe):  Pandas dataframe containing the data to run the tests on
            target_var (String):  The name of the target variable on which to run the tests (should be binary for classification)
            k_features (int):  the number of features to return e.g. the top X correlated variables
            random_state (int):  random number seed
        
        """
        self.df = df
        self.target_var = target_var
        self.k_features = k_features
        self.random_state = random_state
        
        
    def squared_corr(self):
        """ Method for calculating the top X variables with the highest squared correlation with the target variable
        
        Args:
        None
        
        Attributes:
        None
        
        Returns:
        Pandas Dataframe with the top X variables related to the target by squared correlation
        """
        # Get all numeric data into a dataframe
        corr_df_input = self.df._get_numeric_data()

        # Calculate the squared correlation and remove the target
        squared_corr = pd.DataFrame(corr_df_input[corr_df_input.columns[1:]].corr()[self.target_var][:]).reset_index()
        squared_corr.columns = ('Variable', 'Squared_Correlation')
        squared_corr.loc[:,'Squared_Correlation'] = squared_corr[['Squared_Correlation']]**2
        squared_corr = squared_corr[squared_corr.Variable != self.target_var]

        # Order and select the top X
        squared_corr.sort_values(by = 'Squared_Correlation', ascending = False, inplace = True)
        squared_corr = squared_corr.iloc[:self.k_features]
    
        return squared_corr
    
    def rf_imp_rank(self, RandomForestClassifier):
  
    """ Method for calculating the top X variables with the highest rf importance with the target variable
        
    Args:
    df (Dataframe)
    target_var (String)
    k_features (Integer)
    RandomForestClassifier (Class)
    
    Attributes:
    df:  Pandas dataframe containing the target and feature variables
    target_var:  The target variable
    k_features:  The number of features to return
    RandomForestClassifier:  A Random Forest classifier using sci-kit learn RandomForestClassifier
        
    Returns:
    Pandas Dataframe with the top X variables by RF Importance 
    """

    cat_features = self.df.loc[:, self.dtypes == object]

    if not cat_features.empty:
        cat_features_transform = pd.get_dummies(cat_features)

        # Append back the numeric variables to get an encoded dataframe
        numeric_features = self.df._get_numeric_data()

        df_encoded = pd.concat([cat_features_transform, numeric_features], axis = 1)

    X = df_encoded.drop([target_var], axis=1)
    y = df_encoded[target_var]
    feat_labels = pd.DataFrame(X.columns)

    # Run the random forest model
    RandomForestClassifier.fit(X, y)

    # Get the rf importance and append the feature variable labels
    importance = pd.DataFrame(forest.feature_importances_)
    rf_importance = feat_labels.merge(importance,left_index = True, right_index = True)
    rf_importance.columns = ['features','rf_importance']
    rf_importance.sort_values('rf_importance', ascending = False, inplace = True) 
    rf_importance['rf_rank'] = range(1, len(rf_importance) + 1)

    rf_importance = rf_importance[rf_importance.rf_rank <= k_features]

    return rf_importance
        
        