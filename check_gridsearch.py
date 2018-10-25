from sklearn import datasets  # サンプル用のデータ・セット
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC  # SVM の実行関数
from sklearn.cross_validation import train_test_split  # 訓練データとテストデータを分ける関数
from sklearn.metrics import classification_report, confusion_matrix  # 学習結果要約用関数

# サンプル用のデータを読み込み
digits = datasets.load_digits()
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


# 探索するパラメータを設定
param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

# 評価関数を指定
scores = ['accuracy', 'precision', 'recall']

# 各評価関数ごとにグリッドサーチを行う
for score in scores:
    print score
    clf = GridSearchCV(SVC(C=1), param_grid, cv=5, scoring=score, n_jobs=-1)  # n_jobs: 並列計算を行う（-1 とすれば使用PCで可能な最適数の並列処理を行う）
    clf.fit(X_train, y_train)

    print clf.best_estimator_  # 最適なパラメータを表示

    for params, mean_score, all_scores in clf.grid_scores_:
        print "{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params)

    # 最適なパラメータのモデルでクラスタリングを行う
    y_true, y_pred = y_test, clf.predict(X_test)
    print classification_report(y_true, y_pred)  # クラスタリング結果を表示
    print confusion_matrix(y_true, y_pred)       # クラスタリング結果を表示

