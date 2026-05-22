# The models used

In this ML benchmark puplication, various ML models have been in-build along with the benchmark itself to serve as a point of comparison for any custom models and to also just test out the benchmark without any custom ML model


```python
"Logistic Regression": (
    make_preprocessing_pipeline(
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=cfg.random_seed), cfg
    ),
    {
        "kbest__k": [50, 100, 200],
        "model__C": [0.01, 0.1, 1, 10],
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs"],
    },
),
"SVM": (
    make_preprocessing_pipeline(
        SVC(probability=True, class_weight="balanced", random_state=cfg.random_seed), cfg
    ),
    {
        "kbest__k": [50, 100, 200],
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale"],
    },
),
"Random Forest": (
    make_preprocessing_pipeline(
        RandomForestClassifier(class_weight="balanced", random_state=cfg.random_seed), cfg
    ),
    {
        "kbest__k": [100, 200],
        "model__n_estimators": [300, 600],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 3, 5],
    },
),
"Gradient Boosting": (
    make_preprocessing_pipeline(GradientBoostingClassifier(random_state=cfg.random_seed), cfg),
    {
        "kbest__k": [100, 200],
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.03, 0.1],
        "model__max_depth": [2, 3],
    },
),
"KNN": (
    make_preprocessing_pipeline(KNeighborsClassifier(), cfg),
    {
        "kbest__k": [50, 100, 200],
        "model__n_neighbors": [3, 5, 7, 11],
        "model__weights": ["uniform", "distance"],
    },
),
"Naive Bayes": (
    make_preprocessing_pipeline(GaussianNB(), cfg),
    {
        "kbest__k": [50, 100, 200],
        "model__var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7],
    },
),
"Neural Network": (
    make_preprocessing_pipeline(MLPClassifier(max_iter=2000, random_state=cfg.random_seed), cfg),
    {
        "kbest__k": [50, 100, 200],
        "model__hidden_layer_sizes": [(50,), (100,), (100, 50)],
        "model__alpha": [1e-4, 1e-3, 1e-2],
        "model__learning_rate": ["constant", "adaptive"],
    },
),
```