import os
import pickle
from sklearn import feature_extraction, naive_bayes, tree, metrics


def transform_input_data(input_training, input_test):
    vectorized = feature_extraction.text.HashingVectorizer(
        stop_words="english", binary=True
    )  # n_features
    return (
        vectorized.fit_transform(input_training),
        vectorized.fit_transform(input_test),
    )


def get_data(filename):
    data = pickle.load(open(filename, "rb"))
    x_train, x_test, y_train, y_test = (
        data["x_train"],
        data["x_test"],
        data["y_train"],
        data["y_test"],
    )
    x_train_transform, x_test_transform = transform_input_data(x_train, x_test)
    return x_train_transform, x_test_transform, y_train, y_test


def fit_data(x, y, classifier):
    return classifier.fit(x, y)


def predict_data(test_data, classifier):
    return classifier.predict(test_data)


def accuracy_score(prediction, test):
    return metrics.accuracy_score(prediction, test)


def classify_and_predict(
    x_train_transform, x_test_transform, y_train, y_test, classifier
):
    fit_data(x_train_transform, y_train, classifier)
    prediction = predict_data(x_test_transform, classifier)
    return accuracy_score(prediction, y_test)


def get_full_path(rel_path):
    return os.path.join(os.path.dirname(__file__), rel_path)


def main():
    sklearn_data = "data/sklearn-data.pickle"
    data = get_data(get_full_path(sklearn_data))

    naive_bayes_classifier = naive_bayes.BernoulliNB()
    decision_tree_classifier = tree.DecisionTreeClassifier()

    naive_bayes_accuracy = classify_and_predict(*data, naive_bayes_classifier)
    print("Accuracy Naive Bayes: {}".format(naive_bayes_accuracy))

    decision_tree_accuracy = classify_and_predict(*data, decision_tree_classifier)
    print("Accuracy Decision Tree: {}".format(decision_tree_accuracy))


if __name__ == "__main__":
    main()
