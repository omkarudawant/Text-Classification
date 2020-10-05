from sklearn.pipeline import Pipeline as Sklearn_pipeline
from imblearn.pipeline import Pipeline as Imblearn_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import processors

smt = SMOTE(random_state=0)

preprocess_pipeline = Sklearn_pipeline(
    [
        (
            'drop_duplicates', processors.DropDuplicates()
        ),
        (
            'remove_short_sentences', processors.RemoveShortSentences(
                column_name='short_descriptions',
                min_sentence_length=2
            )
        ),
        (
            'filer_labels', processors.FilterLabels(
                min_label_datapoints=500,
                label_column_name='assignment_groups'
            )
        ),
        (
            'clean_text', processors.CleanText(
                column_name='short_descriptions'
            )
        ),
    ]
)

train_pipeline = Imblearn_pipeline(
    [
        ('count_vec', CountVectorizer()),
        ('over_sample_smote', smt),
        ('naive_bayes', MultinomialNB())

    ]
)
