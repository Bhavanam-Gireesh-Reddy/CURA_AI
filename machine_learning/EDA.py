import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")


BASE = os.path.dirname(os.path.abspath(__file__))

DATA = os.path.join(BASE,"data","indian_liver_patient.csv")

ART = os.path.join(BASE,"artifacts")
PLOT = os.path.join(ART,"plots")
REPORT = os.path.join(ART,"reports")

os.makedirs(PLOT,exist_ok=True)
os.makedirs(REPORT,exist_ok=True)


TARGET="Dataset"



def load():
    return pd.read_csv(DATA)



def clean(df):

    df=df.copy()

    df["Gender"]=df["Gender"].map(
        {"Male":1,"Female":0}
    )

    df[TARGET]=df[TARGET].map(
        {1:1,2:0}
    )

    df.drop_duplicates(inplace=True)

    df.fillna(
        df.median(numeric_only=True),
        inplace=True
    )

    return df



def features(df):

    df["Indirect_Bilirubin"] = (
        df.Total_Bilirubin-
        df.Direct_Bilirubin
    )


    df["AST_ALT_Ratio"] = (
        df.Aspartate_Aminotransferase /
        (df.Alamine_Aminotransferase+1)
    )


    df["Globulin"] = (
        df.Total_Protiens-
        df.Albumin
    )


    df["Age_Group"]=pd.cut(
        df.Age,
        [0,30,45,60,100],
        labels=[0,1,2,3]
    ).astype(int)


    for c in [
        "Total_Bilirubin",
        "Direct_Bilirubin",
        "Alkaline_Phosphotase",
        "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase"
    ]:
        df["log_"+c]=np.log1p(df[c])


    return df




def eda(df):

    df.describe().T.to_csv(
        REPORT+"/summary.csv"
    )


    df[TARGET].value_counts().plot(
        kind="bar"
    )

    plt.title("Liver Disease Distribution")
    plt.savefig(
        PLOT+"/target.png"
    )

    plt.close()


    df.corr().to_csv(
        REPORT+"/correlation.csv"
    )




def vif_remove(df):

    X=df.drop(
        columns=[TARGET]
    )

    X=X.select_dtypes(
        include=np.number
    )


    while True:

        vif=pd.Series(
            [
            variance_inflation_factor(
                X.values,i
            )
            for i in range(
                X.shape[1]
            )
            ],
            index=X.columns
        )


        if vif.max()>5:

            X.drop(
                columns=[
                vif.idxmax()
                ],
                inplace=True
            )

        else:
            break


    return pd.concat(
        [X,df[TARGET]],
        axis=1
    )




def main():

    df=load()

    df=clean(df)

    df=features(df)

    eda(df)

    df=vif_remove(df)


    df.to_csv(
        ART+"/processed_liver_data.csv",
        index=False
    )


    print("EDA Completed")
    print(df.shape)



if __name__=="__main__":
    main()