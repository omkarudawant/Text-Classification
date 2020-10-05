from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

import pandas as pd
import re
import string
import spacy

nlp = spacy.load('en_core_web_sm')


def display_metrics(true, pred):
    f1 = round(f1_score(y_true=true, y_pred=pred, average='weighted') * 100, 2)
    precision = round(
        precision_score(y_true=true, y_pred=pred, average='weighted') * 100, 2)
    recall = round(
        recall_score(y_true=true, y_pred=pred, average='weighted') * 100, 2)
    acc = round(accuracy_score(y_true=true, y_pred=pred) * 100, 2)

    print(
        f'Accuracy: {acc} | F1: {f1} | Precision: {precision} | Recall: '
        f'{recall}')

    return acc, f1, precision, recall


class DropDuplicates(BaseEstimator, TransformerMixin):
    def __init__(self):
        super(DropDuplicates, self).__init__()

    # noinspection PyMethodMayBeStatic
    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        dataframe = df.copy()
        print('Entered DropDuplicates')
        dataframe = dataframe[~dataframe.duplicated()]
        dataframe.dropna(inplace=True)
        print('Done drop_duplicates')
        return dataframe

    def fit(self, X, y) -> 'DropDuplicates':
        return self

    def transform(self, X):
        X = X.copy()
        dropped_duplicates = self.drop_duplicates(df=X)
        X = dropped_duplicates.copy()
        return X


class RemoveShortSentences(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str, min_sentence_length: int):
        super(RemoveShortSentences, self).__init__()
        self.min_sentence_length = min_sentence_length
        self.column_name = column_name

    def remove_short_sentences(self,
                               df: pd.DataFrame,
                               min_length: int,
                               col_name: str) -> pd.DataFrame:
        dataframe = df.copy()
        tqdm.pandas()
        dataframe[self.column_name] = dataframe[self.column_name][
            dataframe[self.column_name].progress_apply(
                lambda x: len(x.split()) > self.min_sentence_length
            )]
        dataframe.dropna(inplace=True)

        return dataframe

    def fit(self, X, y) -> 'RemoveShortSentences':
        return self

    def transform(self, X):
        X = X.copy()
        print(f'Entered RemoveShortSentences: {X.shape}')
        removed_short_sentences = self.remove_short_sentences(
            df=X,
            min_length=self.min_sentence_length,
            col_name=self.column_name
        )
        X = removed_short_sentences.copy()
        print('Done RemoveShortSentences')
        return X


class FilterLabels(BaseEstimator, TransformerMixin):
    def __init__(self, min_label_datapoints: int, label_column_name: str):
        super(FilterLabels, self).__init__()
        self.min_label_datapoints = min_label_datapoints
        self.label_column_name = label_column_name

    def find_max_label(self, df: pd.DataFrame) -> tuple:
        dataframe = df.copy()

        counts = dataframe[self.label_column_name].value_counts()
        max_label_name = counts.index[0]
        required_datapoints = counts[1]

        print(
            f'Required dp: {required_datapoints} | MaxLabelName: '
            f'{max_label_name}')

        max_class = dataframe[
            dataframe[self.label_column_name] == max_label_name]

        max_class = max_class.sample(required_datapoints)
        return max_class, max_label_name

    def filter_labels(self, df: pd.DataFrame):
        dataframe = df.copy()

        max_class, max_label_name = self.find_max_label(df=dataframe)

        print(f'Max class:\n{max_class}')

        dataframe = dataframe[
            ~dataframe[self.label_column_name].isin([max_label_name])
        ]

        counts = dataframe[self.label_column_name].value_counts()
        required_labels = counts[counts > self.min_label_datapoints]

        dataframe = dataframe[
            dataframe[self.label_column_name].isin(
                required_labels.index.tolist())
        ]

        result_dataframe = dataframe.append(max_class)
        result_dataframe = result_dataframe.sample(frac=1)
        print(f'Filtered dataframe: {result_dataframe.shape}')

        return result_dataframe

    def fit(self, X, y) -> 'FilterLabels':
        return self

    def transform(self, X):
        X = X.copy()
        print(f'Entered FilterLabels: {X.shape}')
        filter_labels = self.filter_labels(df=X)
        X = filter_labels.copy()

        # X['assignment_groups'].value_counts().plot(kind='bar',
        #                                                 figsize=(15, 5))
        # plt.show()

        print('Done FilterLabels')
        return X


class CleanText(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        super(CleanText, self).__init__()
        self.column_name = column_name

    # noinspection PyMethodMayBeStatic
    def _clean_text(self, text: str):
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        return text

    # noinspection PyMethodMayBeStatic
    def preprocess_text(self, text: str):
        sentence = list()
        doc = nlp(text)
        for word in doc:
            sentence.append(word.lemma_)
        return ' '.join(sentence)

    def clean_text(self, df: pd.DataFrame):
        dataframe = df.copy()
        print(dataframe.isna().sum() / len(dataframe) * 100)
        tqdm.pandas()
        dataframe[self.column_name] = dataframe[
            self.column_name].progress_apply(lambda x: self._clean_text(x))

        # dataframe[self.column_name] = dataframe[
        #     self.column_name].progress_apply(lambda x:
        #     self.preprocess_text(x))

        dataframe[self.column_name] = dataframe[self.column_name].str.replace(
            '-PRON-', '')

        dataframe = dataframe.sample(frac=1)

        return dataframe

    def fit(self, X, y) -> 'CleanText':
        return self

    def transform(self, X):
        X = X.copy()
        print(f'Entered CleanText: {X.shape}')
        cleaned_dataframe = self.clean_text(df=X)
        X = cleaned_dataframe.copy()
        print('Done CleanText')
        return X

# class OverSample(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         super(OverSample, self).__init__()
#         self.smt = SMOTE(random_state=0)
#         self.vectorizer = CountVectorizer()
#         self.encoder = LabelEncoder()
#
#     def fit(self, X, y):
#         print(f'Entered OverSample')
#         X_vec = self.vectorizer.fit_transform(X)
#         y_vec = self.encoder.fit_transform(y)
#         X, y = self.smt.fit_resample(X_vec, y_vec)
#         X = self.vectorizer.transform(X)
#         y = self.encoder.transform(y)
#         print('Done OverSample')
#         return self
#
#     def transform(self, X):
#         X = self.vectorizer.transform(X)
#         return X


# class ExtractFromArrays(BaseEstimator, TransformerMixin):
#     def __init__(self, column_name):
#         super(ExtractFromArrays, self).__init__()
#         self.column_name = column_name
#
#     # noinspection PyMethodMayBeStatic
#     def extract(self, text):
#         return ' '.join([t for t in text])
#
#     def extract_from_array(self, df: pd.DataFrame):
#         print(f'Entered ExtractFromArrays: {df.shape}')
#         dataframe = df.copy()
#         dataframe[self.column_name] = dataframe[self.column_name].apply(
#             lambda x: self.extract(x))
#         print('Done ExtractFromArrays')
#         return dataframe
#
#     def fit(self, X, y) -> 'ExtractFromArrays':
#         X = X.copy()
#         extracted_text_from_arrays = self.extract_from_array(df=X)
#         X = extracted_text_from_arrays.copy()
#         return self
#
#     def transform(self, X):
#         return self.extract_from_array(df=X.copy())
