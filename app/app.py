from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from io import BytesIO

from pipeline import preprocess_pipeline
from processors import display_metrics
from pipeline import train_pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import uvicorn
import joblib
import pandas as pd

app = FastAPI()


@app.post('/preprocess/')
def preprocess_data(file: UploadFile = File(...)):
    if file.filename.endswith(('.csv', '.CSV')):
        csv_file = file.file.read()
        df = pd.read_csv(BytesIO(csv_file))
        output = preprocess_pipeline.transform(X=df)

        output.to_csv('../data/preprocessed_data.csv', index=False)
        return {
            "Message": "Preprocessed data save to "
                       "../data/preprocessed_data.csv"
        }

    else:
        return {"Message": "Please upload a csv file"}


@app.post('/train/')
def train_model(file: UploadFile = File(...)):
    if file.filename.endswith(('.csv', '.CSV')):
        csv_file = file.file.read()
        df = pd.read_csv(BytesIO(csv_file))

        X = df['short_descriptions']
        y = df['assignment_groups']

        enc = LabelEncoder()
        y_enc = enc.fit_transform(y)
        joblib.dump(value=enc, filename='../models/encoder.joblib')

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.1,
            random_state=0
        )

        X_test, X_val, y_test, y_val = train_test_split(
            X_test, y_test,
            test_size=0.2,
            random_state=0
        )

        print(
            X_train.shape, y_train.shape,
            X_test.shape, y_test.shape,
            X_val.shape, y_val.shape
        )

        val_df = pd.DataFrame(
            {
                'short_descriptions': X_val,
                'assignment_groups': y_val
            }
        ).to_csv('../data/validation_set.csv', index=False)

        predictor = train_pipeline.fit(X=X_train, y=y_train)

        y_hat = predictor.predict(X_test)

        acc, f1, precision, recall = display_metrics(true=y_test, pred=y_hat)

        joblib.dump(
            value=predictor,
            filename='../models/model_pipeline.joblib',
            compress=2
        )

        report = classification_report(y_true=y_test,
                                       y_pred=predictor.predict(X_test),
                                       output_dict=True)
        report_df = pd.DataFrame(report)

        report_df.T.to_csv('../data/classification_report.csv')

        return {
            "Accuracy": acc,
            "F1": f1,
            "Precision": precision,
            "Recall": recall
        }

    else:
        return {"Message": "Please upload a csv file"}


class ShortDescription(BaseModel):
    description: str


@app.post('/predict_single/')
async def predict_asgn_group(short_desc: ShortDescription):
    short_desc_dict = short_desc.dict()
    model = joblib.load('../models/model_pipeline.joblib')
    if short_desc.description:
        prediction = model.predict([short_desc.description])
        print(prediction)

        prediction_enc = prediction[0]
        short_desc_dict.update({
            "Predicted assignment group": str(prediction_enc)
        })

    return short_desc_dict


@app.post('/predict_batch/')
async def batch_predictions(file: UploadFile = File(...)):
    model = joblib.load('../models/model_pipeline.joblib')
    try:
        if file.filename.endswith(('.csv', '.CSV')):
            csv_file = await file.read()
            df = pd.read_csv(BytesIO(csv_file))
            df.dropna(inplace=True)

            X = df['short_descriptions'].values
            y = df['assignment_groups'].values

            pred_asgn_grps = model.predict(X)
            pred_asgn_grps_enc = pred_asgn_grps

            output = list()
            for short_desc, true_grp, pred_grp in zip(X[:50],
                                                      y[:50],
                                                      pred_asgn_grps_enc[:50]):
                result = dict()
                result['short_descriptions'] = short_desc
                result['true_assignment_groups'] = true_grp
                result['pred_assignment_groups'] = pred_grp
                output.append(result)

            return output

        else:
            return {"Error": "Please upload a csv file"}

    except Exception as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    uvicorn.run(app, port=5002)
