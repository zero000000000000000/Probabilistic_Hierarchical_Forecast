import pandas as pd
import os
import numpy as np
import math

def Validate_variance(mat,x,y):
    '''
    Check the noise law
    '''
    Vsum = np.sum(mat)
    VA = np.sum(mat[0:2,0:2]) + y
    VB = np.sum(mat[2:,2:]) + y
    VAA = mat[0][0] + x + y*0.25
    VAB = mat[1][1] + x + y*0.25
    VBA = mat[2][2] + x + y*0.25
    VBB = mat[3][3] + x + y*0.25

    print(Vsum,VA,VB,VAA,VAB,VBA,VBB)

    if (Vsum < VA) & (Vsum < VB) & (VAA > VA) & (VAB > VA) & (VBA > VB) & (VBB > VB):
        return True
    else:
        return False

def Generate_Hierarchical_Data(df):
    '''
    Generate the whole hierarchy by summing
    '''
    T = df.iloc[:,0] + df.iloc[:,1] + df.iloc[:,2] + df.iloc[:,3]
    A = df.iloc[:,0] + df.iloc[:,1]
    B = df.iloc[:,2] + df.iloc[:,3]
    new_df = pd.DataFrame({'T':T,'A':A,'B':B,
                           'AA':df.iloc[:,0],
                           'AB':df.iloc[:,1],
                           'BA':df.iloc[:,2],
                           'BB':df.iloc[:,3]})
    return new_df

# Load the bottom level data
#print(os.getcwd())
#os.chdir('./Simulation')
print(os.getcwd())
df = pd.read_csv('./Data/Generate_Bottom_Level.csv')
print(df.head(10))

# First type without noise aggregation 
simulate_data_1 = Generate_Hierarchical_Data(df)

# Second type with noise aggregation 
# Set the variance
bottom_err_cov = np.array([[5,3,2,1],[3,4,2,1],[2,2,5,3],[1,1,3,4]])
N = 2000
sigmas1 = 36
sigmas2 = 30

# Validate the variance
#print(Validate_variance(bottom_err_cov,sigmas1,sigmas2))
if Validate_variance(bottom_err_cov,sigmas1,sigmas2):
    # Set seed
    np.random.seed(10)
    Ut = np.random.normal(0,math.sqrt(sigmas1),N).tolist()
    Vt = np.random.normal(0,math.sqrt(sigmas2),N).tolist()
    # Add noise
    df.iloc[:,0] = df.iloc[:,0].add([-1*x - 0.5*y for x,y in zip(Ut,Vt)])
    df.iloc[:,1] = df.iloc[:,1].add([x - 0.5*y for x,y in zip(Ut,Vt)])
    df.iloc[:,2] = df.iloc[:,2].add([-1*x + 0.5*y for x,y in zip(Ut,Vt)])
    df.iloc[:,3] = df.iloc[:,3].add([x + 0.5*y for x,y in zip(Ut,Vt)])
    simulate_data_2 = Generate_Hierarchical_Data(df)
    df_2 = pd.DataFrame({'Ut':Ut,'Vt':Vt})
    df_2.to_csv('./Data/Added_Noise_Data.csv',index=False)

# Save the two type of data
simulate_data_1.to_csv('./Data/Simulated_Data_Without_Added_Noise.csv',index=False)
simulate_data_2.to_csv('./Data/Simulated_Data_With_Added_Noise.csv',index=False)