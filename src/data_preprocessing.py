from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_data(input_df):
    scaler = StandardScaler()
    scaled_data=scaler.fit_transform(input_df).reshape((1,60,5))
    return (scaled_data,scaler)


    
    

    