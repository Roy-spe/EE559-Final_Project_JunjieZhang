from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def compute_classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    return acc, macro_f1, cm