## Movie-Recsys
Система рекомендаций фильмов с использованием контентной и коллаборативной фильтрации. Python, Pandas, Scikit-learn, Scikit-surprise, FAISS.

## Содержание

- [О проекте](#о-проекте)
- [Основные возможности](#основные-возможности)
- [Технологии и библиотеки](#технологии-и-библиотеки)
- [Данные](#данные)
- [Структура проекта](#структура-проекта)
- [Примеры использования](#примеры-использования)
- [Метрики качества](#метрики-качества)

---

## О проекте

Данный проект представляет собой комплексную систему рекомендаций фильмов, реализованную на языке Python. Проект демонстрирует полный цикл работы с данными: от предобработки и исследования данных (EDA) до построения и оценки нескольких моделей рекомендаций.

Система объединяет три классических подхода к построению рекомендательных систем, что позволяет сравнить их эффективность и применимость в различных сценариях.

---

## Основные возможности

Проект включает три основных типа рекомендательных систем.

<table>
<tr>
<td width="33%" valign="top">

### 1. Baseline

**Простая рекомендательная система**

- Рекомендует фильмы на основе рейтинга и популярности
- Использует взвешенный рейтинг (Weighted Rating)
- Позволяет получить топ-фильмы в разрезе конкретного жанра

</td>
<td width="33%" valign="top">

### 2. Content-Based

**Контентная фильтрация**

- Простая версия: по описанию, жанрам, ключевым словам
- Улучшенная версия: с учетом актеров и режиссера
- Векторизация: TF-IDF, CountVectorizer
- Оптимизация поиска: FAISS

</td>
<td width="33%" valign="top">

### 3. Collaborative

**Коллаборативная фильтрация**

- Предсказывает рейтинг на основе оценок других пользователей
- Алгоритм SVD (Singular Value Decomposition)
- Библиотека scikit-surprise
- Метрики: RMSE, MAE

</td>
</tr>
</table>

---

## Технологии и библиотеки

<table>
<tr>
<td valign="top" width="50%">

**Обработка данных**

![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

**Машинное обучение**

![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![scikit-surprise](https://img.shields.io/badge/scikit--surprise-1.1.3-blue?style=flat-square)

</td>
<td valign="top" width="50%">

**Поиск ближайших соседей**

![FAISS](https://img.shields.io/badge/FAISS-cpu-blueviolet?style=flat-square)

**Визуализация**

![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat-square)

**Среда разработки**

![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat-square&logo=google-colab&logoColor=white)

</td>
</tr>
</table>

---

## Данные

Проект использует четыре датасета, которые должны находиться в корневой папке проекта.

| Файл | Описание |
|------|----------|
| `movies_metadata.csv` | Основная информация о фильмах (бюджет, жанры, описание и т.д.) |
| `keywords.csv` | Ключевые слова, связанные с фильмами |
| `credits.csv` | Информация об актерах и съемочной группе |
| `ratings.csv` | Оценки пользователей для фильмов (используется для коллаборативной фильтрации) |

---

## Структура проекта

Проект состоит из одного основного Jupyter Notebook, который логически разделен на следующие этапы.

### Этап 1. Загрузка и предобработка данных

- Установка и импорт библиотек
- Загрузка CSV-файлов
- Очистка данных: обработка пропущенных значений, удаление некорректных строк, объединение таблиц
- Перевод названий фильмов на английский язык
- Извлечение признаков из JSON-подобных столбцов (жанры, ключевые слова, актеры, режиссер)

### Этап 2. Исследовательский анализ данных (EDA)

- Визуализация распределения средних оценок фильмов
- Визуализация распределения жанров
- Построение матрицы корреляций для числовых признаков
- Анализ распределения оценок пользователей

### Этап 3. Построение рекомендательных систем

- **Baseline:** реализация функции `simple_recommender` на основе взвешенного рейтинга
- **Content-Based:** создание двух версий рекомендаций на основе сходства контента. Оптимизация поиска с помощью FAISS
- **Collaborative:** обучение и оценка SVD-модели с помощью `scikit-surprise`. Реализация функции `predict_rating`

---

## Примеры использования

### Пример 1. Рекомендации на основе популярности (для жанра Animation)

```python
top_animation_movies = simple_recommender(movies, genre='Animation')
print(top_animation_movies.head())
```

**Результат:**

```
                   title  vote_count  vote_average  popularity        wr
359        The Lion King        5520             8   21.605761  7.585763
5505       Spirited Away        3968             8   41.048867  7.464227
9746  Howl's Moving Castle        2049             8   16.136048  7.159212
2899     Princess Mononoke        2041             8   17.166725  7.157211
5863    My Neighbor Totoro        1730             8   13.507299  7.071325
```

### Пример 2. Рекомендации на основе контента (для The Lion King)

```python
print("Рекомендации для 'The Lion King':")
recommendations = get_recommendations('The Lion King', cosine_sim=cosine_sim_tfidf, top_n=5)
print(recommendations)
```

**Результат:**

```
Рекомендации для 'The Lion King':
                             title                          genres  similarity
651      James and the Giant Peach  [adventure, animation, family]       0.321
270               Man of the House        [action, comedy, family]       0.315
1002                    Cinderella    [family, fantasy, animation]       0.314
2238    I'll Be Home for Christmas                [comedy, family]       0.309
32738                    Flashback               [horror, foreign]       0.304
```

### Пример 3. Предсказание рейтинга (для пользователя 1 и фильма 302)

```python
predicted_rating = predict_rating(1, 302)
print(f"Предсказанный рейтинг: {predicted_rating:.2f}")
```

**Результат:**

```
Предсказанный рейтинг: 3.84
```

---

## Метрики качества

Для оценки коллаборативной фильтрации используются следующие метрики:

| Метрика | Значение |
|---------|----------|
| **RMSE** (Root Mean Squared Error) | 0.8685 |
| **MAE** (Mean Absolute Error) | 0.6662 |

> Значения усреднены по 5 фолдам кросс-валидации.

---

