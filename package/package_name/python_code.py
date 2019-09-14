class VarSelection:
    """ The variable selection class contains a set of methods to support variable selection in classification modeling """
    
    def __init__(k_features = 10, test_size = 0.25, random_state = 1):
        
        """ Method for initializing a VarSelection object
        
        Args:
        k_features (int)
        test_size (float)
        random_state (int)
        
        Attributes:
            k_features (int):  the number of features to return e.g. the top X correlated variables
            test_size (float):  the % of the dataframe to use as the test
            random_state (int):  random number seed
        """
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
        
    