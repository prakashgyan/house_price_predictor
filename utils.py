import pandas as pd
import random




def handle_missing_data(df):
    """This function handles all types of missing data form the dataframe."""
    
    # We will replace the NA values in this feature as None which mean no Pool
    df["PoolQC"] = df["PoolQC"].fillna("None")
    # MiscFeature : data description says NA means "no misc feature"
    df["MiscFeature"] = df["MiscFeature"].fillna("None")
    # Alley : data description says NA means "no alley access"
    df["Alley"] = df["Alley"].fillna("None")
    # Fence : data description says NA means "no fence"
    df["Fence"] = df["Fence"].fillna("None")
    # FireplaceQu : data description says NA means "no fireplace"
    df["FireplaceQu"] = df["FireplaceQu"].fillna("None")
    # LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.
    # Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))
    # GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        df[col] = df[col].fillna('None')
    # GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        df[col] = df[col].fillna(0)
    # BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for having no basement
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        df[col] = df[col].fillna(0)
    # BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these categorical basement-related features, NaN means that there is no basement.
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        df[col] = df[col].fillna('None')
    # MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.
    df["MasVnrType"] = df["MasVnrType"].fillna("None")
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    # MSZoning (The general zoning classification) : 'RL' is by far the most common value. So we can fill in missing values with 'RL'
    df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
    # Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.
    df = df.drop(['Utilities'], axis=1)
    # Functional : data description says NA means typical
    df["Functional"] = df["Functional"].fillna("Typ")
    # Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    # KitchenQual: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.
    df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
    # Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
    df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
    df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
    # SaleType : Fill in again with most frequent which is "WD"
    df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
    # MSSubClass : Na most likely means No building class. We can replace missing values with None
    df['MSSubClass'] = df['MSSubClass'].fillna("None")
    
    return df



def transform_num_to_cat(df):
    #MSSubClass=The building class
    df['MSSubClass'] = df['MSSubClass'].apply(str)


    #Changing OverallCond into a categorical variable
    df['OverallCond'] = df['OverallCond'].astype(str)


    #Year and month sold are transformed into categorical features.
    df['YrSold'] = df['YrSold'].astype(str)
    df['MoSold'] = df['MoSold'].astype(str)

    return df

def add_total_sqft(df):
    # Adding total sqfootage feature 
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    return df


def process_raw_input(df):
    df = handle_missing_data(df)
    df = transform_num_to_cat(df)
    df = add_total_sqft(df)
    df = pd.get_dummies(df)
    print(df)
    return df


def load_model():

    return DummyModel()


