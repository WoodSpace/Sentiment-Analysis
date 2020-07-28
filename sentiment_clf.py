from sklearn.externals import joblib


class SentimentClassifier(object):
    def __init__(self):
        self.model = joblib.load('sentiment_classifier.pkl')
        self.classes_dict = {0: 'Негативный отзыв', 1: 'Позитивный отзыв', -1: 'Ошибка'}

    def predict_text(self, text):
        try:
            lst = [text]
            return self.model.predict(lst)
        except:
            print("Ошибка")
            return -1

    def predict_list(self, list_of_texts):
        try:
            return self.model.predict(list_of_texts)
        except:
            print("Ошибка")
            return -1

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        return self.classes_dict[int(prediction[0])]
