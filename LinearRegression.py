
class LinearRegression:
    """
    Ordinary least squares Linear Regression.
    
    This class implements the same API as sklearn.linear_model.LinearRegression
    but with a simpler implementation using numpy.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set to False, no intercept
        will be used in calculations (i.e. data is expected to be centered).
    
    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.
    
    n_jobs : int, default=None
        The number of jobs to use for the computation. This parameter is kept for API
        compatibility but is not used in this implementation.
    
    positive : bool, default=False
        When set to True, forces the coefficients to be positive. This parameter is kept
        for API compatibility but is not used in this implementation.
    
    Attributes
    ----------
    coef_ : array of shape (n_features,) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
    
    intercept_ : float or array of shape (n_targets,)
        Independent term in the linear model.
    
    rank_ : int
        Rank of matrix X.
    
    singular_ : array of shape (min(X, y),)
        Singular values of X.
    """
    
    def __init__(self, fit_intercept=True, copy_X=True, n_jobs=None, positive=False):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive
        self.coef_ = None
        self.intercept_ = None
        self.rank_ = None
        self.singular_ = None
    
    def fit(self, X, y):
        """
        Fit linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values
        
        Returns
        -------
        self : returns an instance of self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if self.copy_X:
            X = X.copy()
        
        # Handle 1D arrays
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        if self.fit_intercept:
            # Add column of ones for intercept
            X = np.column_stack([np.ones(n_samples), X])
        
        # Compute the least squares solution using numpy's lstsq
        # This is equivalent to: (X^T X)^(-1) X^T y
        try:
            # Use numpy's lstsq for better numerical stability
            beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            
            if self.fit_intercept:
                self.intercept_ = beta[0]
                self.coef_ = beta[1:]
            else:
                self.intercept_ = 0.0
                self.coef_ = beta
            
            self.rank_ = rank
            self.singular_ = s
            
        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix in linear regression")
        
        return self
    
    def predict(self, X):
        """
        Predict using the linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
        
        Returns
        -------
        C : array of shape (n_samples,)
            Returns predicted values
        """
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before using this method.")
        
        X = np.asarray(X)
        
        # Handle 1D arrays
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.fit_intercept:
            return np.dot(X, self.coef_) + self.intercept_
        else:
            return np.dot(X, self.coef_)
    
    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            True values for X
        
        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y
        """
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before using this method.")
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Handle 1D arrays
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        y_pred = self.predict(X)
        
        # Calculate R^2
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
        ss_res = np.sum((y - y_pred) ** 2, axis=0)
        r2 = 1 - (ss_res / ss_tot)
        
        return r2.mean() if r2.size > 1 else r2.item()