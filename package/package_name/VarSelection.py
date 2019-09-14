class VarSelection:
    """ The variable selection class contains a set of methods to support variable selection in classification / regression modeling """
    
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