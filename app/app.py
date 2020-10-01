from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from io import BytesIO

import uvicorn
import joblib
import numpy as np
import pandas as pd

app = FastAPI()
model = joblib.load('../models/log_reg_tkt')
vectorizer = joblib.load('../models/vectorizer')
encoder = joblib.load('../models/encoder')


class ShortDescription(BaseModel):
    description: str


@app.post('/predict_class/')
async def predict_asgn_group(short_desc: ShortDescription):
    short_desc_dict = short_desc.dict()
    if short_desc.description:
        input_text = np.array([short_desc.description])
        input_text_vec = vectorizer.transform(input_text)
        prediction = model.predict(input_text_vec)
        prediction_enc = encoder.inverse_transform(prediction)[0]
        short_desc_dict.update({
            "Predicted assignment group": str(prediction_enc)
        })

    return short_desc_dict


@app.post('/predict_batch/')
async def batch_predictions(file: UploadFile = File(...)):
    try:
        if file.filename.endswith(('.csv', '.CSV')):
            csv_file = await file.read()
            df = pd.read_csv(BytesIO(csv_file))
            df.dropna(inplace=True)

            X = df['short_descriptions'].values
            y = df['assignment_groups'].values

            X_vec = vectorizer.transform(X)
            pred_asgn_grps = model.predict(X_vec)
            pred_asgn_grps_enc = encoder.inverse_transform(pred_asgn_grps)

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
        return {
            "Error": "Please check if input csv file has "
                     "'short_descriptions' and 'assignment_groups' columns "
                     "in it"
        }


if __name__ == '__main__':
    uvicorn.run(app, port=5002)
