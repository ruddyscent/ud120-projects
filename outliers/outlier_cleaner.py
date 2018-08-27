#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    import numpy as np

    data = np.hstack((ages, net_worths, (net_worths - predictions)**2))
    sorted_data = np.copy(data[data[:,2].argsort()])
    cleaned_data = sorted_data[:int(0.9*data.shape[0])]
    return cleaned_data
