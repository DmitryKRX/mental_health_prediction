<a href="https://colab.research.google.com/drive/1Xaqmymh-_KAgMDQYXCUMvsDWGCpP4iQz">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> 

# Определение депрессии у пациентов на основании клинических данных

Данная тема и набор данных взяты с Kaggle: 
<a href="https://www.kaggle.com/competitions/playground-series-s4e11/">
  <img src="https://github.com/LelouchFR/skill-icons/blob/main/assets/kaggle-auto.svg" alt="Open In Kaggle" width="20" height="20"/>
</a>
 https://www.kaggle.com/competitions/playground-series-s4e11/

## Описание задачи

Требуется использовать данные обследования психического здоровья для предсказания депрессии у пациентов.

Результаты оцениваются с помощью метрики **Accuracy**.

## Описание данных

В тренировочном наборе представлено 20 столбцов, один из которых является таргетом:

**Числовые данные**

- 'id',
- 'Age',
- 'Academic Pressure',
- 'Work Pressure',
- 'CGPA',
- 'Study Satisfaction',
- 'Job Satisfaction',
- 'Work/Study Hours',
- 'Financial Stress',
- 'Depression' — целевая переменная.

**Категориальные данные**

- 'Name',
- 'Gender',
- 'City',
- 'Working Professional or Student',
- 'Profession',
- 'Sleep Duration',
- 'Dietary Habits',
- 'Degree',
- 'Have you ever had suicidal thoughts ?',
- 'Family History of Mental Illness'.

**Распределение целевой переменной в тренировочном наборе**

![image](https://github.com/user-attachments/assets/138cbef1-f740-4d13-b72c-89565200416a)

## Предобработка данных

- Практическив каждом столбце с категориальными данными встречаются некорректные элементы, не относящиеся к характеристике (названию столбца). Каждый некорректный элемент встречается менее 10 раз, следовательно, их можно заменить на "Unknown".
- Данные о продолжительности сна (Sleep Duration) и диетических привычках (Dietary Habits)  могут быть очень полезны при диагностировании депрессии. Однако в данных столбцах очень много категорий, обозначающих примерно одно и то же, но с разными формулировками. Чтобы избавиться от этого, используем функции с регулярными выражениями, пересобирающие данные в 4 категории.
- В числовых данных встречается много пропусков, заменим их на медианное значение.
- От столбцов "id" и "Name" можно избавиться, т.к. они не дают никакой информации о диагнозе.

## Обучение моделей

Для решения задачи будут рассмотрены несколько моделей и их комбинаций. В качестве результатов оценки моделей будут приведены **Public score** и **Private score**, основанные на предсказаниях на файле 'test.csv', предоставленные Кагглом, и результат метрики на небольшом тренировочно-тестовом наборе, представляющим собой 6% данных из файла 'train.csv', не входящих в процесс обучения моделей.

- **Классический CatBoost**
  - Train-test: 0.936982
  - Public score: 0.94136
  - Private score: 0.94027
- **Ансамбль CatBoostCV с голосованием по большинству**
  - **N_folds = 5**
    - Train-test: 0.937811
    - Public score: 0.94173
    - Private score: 0.94031
  - **N_folds = 10**
    - Train-test: 0.936508
    - Public score: 0.94195
    - Private score: 0.94023
  - **N_folds = 20**
    - Train-test: 0.937100
    - Public score: 0.94168
    - Private score: 0.94007
- **Стекинг (CatBoostCV + Нейронная сеть)**
  - N_folds = 5
    - Train-test: 0.938048
    - Public score: 0.94227
    - Private score: 0.94051
  - N_folds = 10
    - Train-test: 0.936271
    - Public score: 0.94200
    - Private score: 0.93908
- **Балансировка классов** *(описание ниже)*
  - **Классический CatBoost + Нейронная сеть**
    - Train-test: 0.938522
    - Public score: 0.94093
    - Private score: 0.94012
  - **Стекинг (CatBoostCV (n_folds=5) + Нейронная сеть)**
    - Train-test: 0.938522
    - Public score: 0.94136
    - Private score: 0.94025
- **Классический LightGBM: Gradient Boosting**
  - Train-test: 0.938048
  - Public score: 0.93987
  - Private score: 0.93903
- **Классический LightGBM: Random Forest**
  - Train-test: 0.930230
  - Public score: 0.93192
  - Private score: 0.93154
- **CV  LightGBM: Gradient Boosting + Dart (n_folds=5)**
  - Train-test: 0.937811
  - Public score: 0.94035
  - Private score: 0.93947
- **Стекинг (CatboostCV + CV LightGBM + NN)**
  - Train-test: 0.928216
  - Public score: 0.93432
  - Private score: 0.93074

В результате самыми <ins>лучшими</ins> моделями оказались: 
- **Стекинг (CatBoostCV на 5 фолдов + Нейронная сеть)** с результатом *Private score: 0.94025*,
- **Ансамбль CatBoostCV на 5 фолдов с голосованием по большинству** с результатом *Private score: 0.94031*,
- **Классический CatBoost** с результатом  *Private score: 0.94027*

![image](https://github.com/user-attachments/assets/2f307e4c-b123-40c6-99d7-010ca0e8dd41)


---

**Балансировка классов**

Так как данные в датасете сильно несбалансированы, можно попробовать сделать балансировку данных следующим способом.

Создадим датасеты с равномерными распределениями классов (т.е. кол-во объектов обоих классов будет одинаковым). В тренировочном+валидационном датасете (train_val) объектов с классом 0 - 23958, а объектов с классом 1 - 108300. 108300 / 23958 ~ 4.5, следовательно, можно создать 4 полных датасета, в котором будут 23958 объектов первого класса и столько же нулевого, и 1 несбалансированный датасет, куда войдут оставшиеся данные. Объекты нулевого класса будем дублировать из train_val датасета (т.е. будет пять датасетов с одинаковыми объектами нулевого класса).

Затем обучим на каждом датасете модель CatBoost/CatBoostCV и применим ансамблирование с нейронной сетью.

---

## Описание файлов

- **s4e11_solution.ipynb** — юпитер-блокнот с выполнением задания;
- **train.csv** — обучающие данные;
- **test.csv** — тестовые данные;
- **processed_train.csv** — предобработанные тренировочные данные.






