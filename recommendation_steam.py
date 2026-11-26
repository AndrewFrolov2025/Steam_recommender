import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import math
import json
import logging
import argparse
import itertools
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

from implicit.als import AlternatingLeastSquares

LOGGER = logging.getLogger("steam")
MODEL_NAMES = ("Popularity", "TF-IDF", "ItemKNN", "ALS")



# Часть 1 - Общие функции


def require_columns(columns, required: list[str], filename: str,) -> None:
    """Идет проверка на наличие обязательных столбцов"""
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"В {filename} отсутствуют столбцы: {missing}. "f"Доступны: {list(columns)}")


def read_header(path: Path) -> list[str]:
    """Читает только заголовок CSV файла."""
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")

    return pd.read_csv(path, nrows=0).columns.tolist()


def clean_id(series: pd.Series) -> pd.Series:
    """Приводит идентификаторы к строковому типу."""
    return series.astype("string").str.strip()


def parse_dates(series: pd.Series) -> pd.Series:
    """Разбирает даты и приводит их к типу UTC."""
    return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")


def parse_boundary(value: str) -> pd.Timestamp:
    """Преобразует аргумент командной строки в временную границу."""
    timestamp = pd.to_datetime(value, utc=True, errors="raise")
    return pd.Timestamp(timestamp)


def parse_boolean(series: pd.Series) -> pd.Series:
    """Распознаёт стандартные варианты булевых значений. Неизвестные значения становятся пропусками."""
    mapping = {"true": True, "1": True, "1.0": True, "yes": True, "false": False, "0": False, "0.0": False, "no": False,}
    return (series.astype("string").str.strip().str.lower().map(mapping).astype("boolean"))


def save_json(path: Path, data: dict) -> None:
    """Сохраняет JSON."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

# Часть 2 - Загрузка файлов CSV

def load_recommendations(path: Path, max_rows: int) -> tuple[pd.DataFrame, dict]:
    """Загружает необходимые столбцы recommendations.csv."""
    columns = read_header(path)

    required = ["user_id", "app_id", "date", "is_recommended"]
    require_columns(columns, required, path.name)

    frame = pd.read_csv(path, usecols=required, dtype={"user_id": "string", "app_id": "string", "is_recommended": "string", "date": "string",}, nrows=None if max_rows == 0 else max_rows,)

    raw_rows = len(frame)

    frame["user_id"] = clean_id(frame["user_id"])
    frame["app_id"] = clean_id(frame["app_id"])
    frame["date"] = parse_dates(frame["date"])
    frame["positive"] = parse_boolean(frame["is_recommended"])

    valid = (frame["user_id"].notna() & frame["app_id"].notna() & frame["user_id"].ne("").fillna(False) & frame["app_id"].ne("").fillna(False) & frame["date"].notna() & frame["positive"].notna())
    frame = frame.loc[valid, ["user_id", "app_id", "date", "positive"]].copy()
    frame["positive"] = frame["positive"].astype(bool)
    valid_before_deduplication = len(frame)

    """ Для пары пользователь–игра сохраняется первое наблюдаемое событие."""
    """Позднейшие повторные записи не переносятся в прошлое."""
    frame = frame.sort_values("date", kind="stable")
    frame = frame.drop_duplicates(subset=["user_id", "app_id"], keep="first").reset_index(drop=True)

    if frame.empty:
        raise ValueError("После очистки recommendations.csv не осталось данных.")

    audit = {
        "recommendations_loaded_rows": int(raw_rows),
        "recommendations_invalid_rows": int(
            raw_rows - valid_before_deduplication
        ),
        "recommendations_duplicate_pairs_removed": int(
            valid_before_deduplication - len(frame)
        ),
        "recommendations_clean_rows": int(len(frame)),
        "recommendations_min_date": frame["date"].min().isoformat(),
        "recommendations_max_date": frame["date"].max().isoformat(),
    }

    return frame, audit


def load_users(path: Path, needed_user_ids: set[str], chunksize: int) -> tuple[pd.DataFrame, dict]:
    """Полностью просматривает users.csv по частям. В памяти сохраняются только пользователи, присутствующие в загруженных взаимодействиях."""
    """products и reviews используются для описания профилей, но не как признаки моделей."""
    columns = read_header(path)
    required = ["user_id", "products", "reviews"]
    require_columns(columns, required, path.name)
    parts = []
    scanned_rows = 0

    for chunk in pd.read_csv(path, usecols=required, dtype={"user_id": "string"}, chunksize=chunksize):
        scanned_rows += len(chunk)
        chunk["user_id"] = clean_id(chunk["user_id"])
        selected = chunk[chunk["user_id"].isin(needed_user_ids)].copy()
        if not selected.empty:
            parts.append(selected)

    if not parts:
        raise ValueError("В users.csv не найдены пользователи из recommendations.csv.")

    users = pd.concat(parts, ignore_index=True)
    if users["user_id"].duplicated().any():
        raise ValueError("В users.csv найдены повторные user_id. Проверьте исходный файл.")

    for column in ["products", "reviews"]:
        users[column] = pd.to_numeric(users[column], errors="coerce")

    audit = {"users_scanned_rows": int(scanned_rows), "users_matched_rows": int(len(users))}

    return users, audit


def load_users(path: Path, needed_user_ids: set[str], chunksize: int) -> tuple[pd.DataFrame, dict]:
    """Полностью просматривает users.csv по частям."""
    columns = read_header(path)

    required = ["user_id", "products", "reviews"]
    require_columns(columns, required, path.name)

    parts = []
    scanned_rows = 0

    for chunk in pd.read_csv(path, usecols=required, dtype={"user_id": "string"}, chunksize=chunksize):
        scanned_rows += len(chunk)
        chunk["user_id"] = clean_id(chunk["user_id"])

        selected = chunk[chunk["user_id"].isin(needed_user_ids)].copy()
        if not selected.empty:
            parts.append(selected)

    if not parts:
        raise ValueError("В users.csv не найдены пользователи из recommendations.csv.")

    users = pd.concat(parts, ignore_index=True)
    if users["user_id"].duplicated().any():
        raise ValueError("В users.csv найдены повторные user_id. Проверьте исходный файл."
        )

    for column in ["products", "reviews"]:
        users[column] = pd.to_numeric(users[column], errors="coerce")

    audit = {"users_scanned_rows": int(scanned_rows), "users_matched_rows": int(len(users))}

    return users, audit


def load_games(path: Path) -> pd.DataFrame:
    """Загружает games.csv и строит текстовое представление игр."""
    columns = read_header(path)
    require_columns(columns, ["app_id", "title"], path.name)

    games = pd.read_csv(path, dtype={"app_id": "string"}, low_memory=False)
    games["app_id"] = clean_id(games["app_id"])

    valid = (games["app_id"].notna() & games["app_id"].ne("").fillna(False))

    games = games.loc[valid].copy()
    if games["app_id"].duplicated().any():
        raise ValueError("В games.csv найдены повторные app_id. Для объединения необходим уникальный идентификатор игры.")

    games["title"] = (games["title"].astype("string").fillna("").str.strip())

    #Название используется дважды, чтобы немного усилить его вклад.
    text = games["title"] + " " + games["title"]

    # В стандартном games.csv жанров и описаний может не быть.
    # Если такие столбцы добавлены, программа использует их.
    optional_text_columns = [
        "genres",
        "tags",
        "categories",
        "developer",
        "publisher",
        "description",
        "short_description",
    ]

    for column in optional_text_columns:
        if column in games.columns:
            part = games[column].astype("string").fillna("")
            text = text + " " + part

    for column in ["win", "mac", "linux"]:
        if column in games.columns:
            available = parse_boolean(games[column]).fillna(False)
            token = np.where(available.to_numpy(dtype=bool),f" platform_{column}", "")
            text = text + pd.Series(token, index=games.index)

    games["content_text"] = (text.astype("string").fillna("").str.replace(r"\s+", " ", regex=True).str.strip())

    # Защита от полностью пустого текста.
    empty_text = games["content_text"].eq("")
    games.loc[empty_text, "content_text"] = ("unknown_game_" + games.loc[empty_text, "app_id"])

    return games.reset_index(drop=True)


# Часть - 3 Границы и обучающая фильтрация


def choose_boundaries(events: pd.DataFrame, train_end: str | None, val_end: str | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Выбирает временные границы. Такие как:

        train: date < train_end
        validation: train_end <= date < val_end
        test: date >= val_end
    """
    if bool(train_end) != bool(val_end):
        raise ValueError("--train-end и --val-end нужно указывать вместе.")

    if train_end and val_end:
        first = parse_boundary(train_end)
        second = parse_boundary(val_end)
    else:
        days = pd.DatetimeIndex(events["date"].dt.floor("D").unique()).sort_values()

        if len(days) < 3:
            raise ValueError("Для временного разбиения требуется минимум три разные даты.")

        first_index = max(1, min(len(days) - 2, int(len(days) * 0.70)))
        second_index = max(first_index + 1, min(len(days) - 1, int(len(days) * 0.85)))

        first = pd.Timestamp(days[first_index])
        second = pd.Timestamp(days[second_index])

    minimum = events["date"].min()
    maximum = events["date"].max()

    if not (minimum < first < second <= maximum):
        raise ValueError(
            "Некорректные временные границы. "
            f"Период данных: {minimum} — {maximum}. "
            "Требуется min_date < train_end < val_end <= max_date."
        )

    return first, second


def k_core_filter(train: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """ Фильтрует train. Убирает редких пользоваетелей и игры"""

    frame = train.copy()
    while True:
        before = len(frame)

        user_counts = frame["user_id"].value_counts()
        item_counts = frame["app_id"].value_counts()

        good_users = user_counts[user_counts >= min_user].index
        good_items = item_counts[item_counts >= min_item].index

        frame = frame[frame["user_id"].isin(good_users) & frame["app_id"].isin(good_items)].copy()

        if frame.empty:
            raise ValueError("После обучающей фильтрации не осталось данных. ")

        if len(frame) == before:
            break

    return frame.reset_index(drop=True)


# Часть - 4 Индексация и построение матрицы (пользователь-игра)


def build_training_matrix(train: pd.DataFrame, games: pd.DataFrame):
    """Создаёт матрицу и согласованные отображения индексов."""
    user_ids = sorted(train["user_id"].unique().tolist())
    item_ids = sorted(train["app_id"].unique().tolist())

    user_to_index = {identifier: index for index, identifier in enumerate(user_ids)}
    item_to_index = {identifier: index for index, identifier in enumerate(item_ids)}

    rows = train["user_id"].map(user_to_index).to_numpy(dtype=np.int32)
    columns = train["app_id"].map(item_to_index).to_numpy(dtype=np.int32)

    matrix = csr_matrix((np.ones(len(train), dtype=np.float32), (rows, columns)), shape=(len(user_ids), len(item_ids)), dtype=np.float32)

    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    matrix.sort_indices()

    catalog = (games.set_index("app_id").loc[item_ids].reset_index())

    return matrix, catalog, user_ids, user_to_index, item_to_index


def build_seen(events: pd.DataFrame, cutoff: pd.Timestamp, user_to_index: dict, item_to_index: dict,
) -> dict[int, set[int]]:
    """Игры, известные пользователю до cutoff. Включаются как положительные, так и отрицательные отзывы."""
    frame = events.loc[events["date"] < cutoff, ["user_id", "app_id"]].copy()

    frame["u"] = frame["user_id"].map(user_to_index)
    frame["i"] = frame["app_id"].map(item_to_index)

    frame = frame.dropna(subset=["u", "i"])

    result = {}

    for user_index, group in frame.groupby("u", sort=False):
        result[int(user_index)] = set(group["i"].astype(int).tolist())

    return result

def build_targets(events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp | None, user_to_index: dict, item_to_index: dict, seen: dict[int, set[int]]) -> tuple[dict[int, set[int]], dict]:
    """Формирует положительные цели для warm-start-оценки."""
    period_mask = (events["date"].ge(start) & events["positive"])

    if end is not None:
        period_mask &= events["date"].lt(end)

    frame = events.loc[period_mask, ["user_id", "app_id"]].copy()

    frame["u"] = frame["user_id"].map(user_to_index)
    frame["i"] = frame["app_id"].map(item_to_index)

    unknown_user = frame["u"].isna()
    unknown_item = frame["i"].isna()

    audit = {
        "positive_rows_in_period": int(len(frame)),
        "rows_with_unknown_user": int(unknown_user.sum()),
        "rows_with_unknown_item": int(unknown_item.sum()),
    }

    # Эти два показателя могут пересекаться: одна строка может иметь и неизвестного пользователя, и неизвестную игру.
    frame = frame[~unknown_user & ~unknown_item].copy()

    targets = {}
    excluded_seen = 0

    for user_index, group in frame.groupby("u", sort=False):
        user_index = int(user_index)

        relevant = set(group["i"].astype(int).tolist())
        known = seen.get(user_index, set())

        excluded_seen += len(relevant & known)
        relevant -= known

        if relevant:
            targets[user_index] = relevant

    audit["known_targets_excluded"] = int(excluded_seen)
    audit["eligible_users"] = int(len(targets))
    audit["eligible_relevant_pairs"] = int(sum(len(items) for items in targets.values()))

    return targets, audit


def sample_users(targets: dict[int, set[int]], limit: int, seed: int) -> list[int]:
    """Воспроизводимо выбирает пользователей для оценки."""
    users = np.array(sorted(targets), dtype=np.int64)
    if len(users) == 0:
        raise ValueError("В периоде не осталось пользователей для оценки. Измените границы дат, фильтры или размер входной выборки.")

    if limit > 0 and len(users) > limit:
        rng = np.random.default_rng(seed)
        users = rng.choice(users, size=limit, replace=False)
        users.sort()

    return users.tolist()

# Часть - 5 Модели


class PopularityModel:
    """Ранжирует игры по числу положительных TRAIN-взаимодействий."""

    def fit(self, matrix: csr_matrix):
        self.values = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)

        return self

    def score(self, user_index: int) -> np.ndarray:
        return self.values.copy()


class ContentModel:
    """TF-IDF-представление игр и профиль обучающей истории."""

    def fit(self, matrix: csr_matrix, catalog: pd.DataFrame, max_features: int):
        self.matrix = matrix

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            sublinear_tf=True,
            max_features=max_features,
            dtype=np.float32,
        )

        self.item_features = self.vectorizer.fit_transform(catalog["content_text"].astype(str)).tocsr()
        self.item_features = normalize(self.item_features, norm="l2", axis=1).tocsr()

        return self

    def score(self, user_index: int) -> np.ndarray:
        profile = (self.matrix.getrow(user_index) @ self.item_features)

        profile = normalize(profile, norm="l2", axis=1)
        values = profile @ self.item_features.T

        return values.toarray().ravel()

    def score_text(self, preferences: str) -> np.ndarray:
        """Оценки для нового пользователя по введённому тексту."""
        profile = self.vectorizer.transform([preferences])

        profile = normalize(profile, norm="l2", axis=1)

        return (profile @ self.item_features.T).toarray().ravel()


class ItemKNNModel:
    """Коллаборативное сходство игр по аудитории пользователей."""

    def fit(self, matrix: csr_matrix, neighbors: int, batch_size: int, jobs: int):
        self.matrix = matrix
        n_items = matrix.shape[1]

        item_vectors = normalize(matrix.T.tocsr(), norm="l2", axis=1).tocsr()

        model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(neighbors + 1, n_items), n_jobs=jobs)

        model.fit(item_vectors)

        row_parts = []
        column_parts = []
        value_parts = []

        for start in range(0, n_items, batch_size):
            stop = min(start + batch_size, n_items)

            distances, indices = model.kneighbors(item_vectors[start:stop], return_distance=True)

            for offset, item_index in enumerate(range(start, stop)):
                similarities = np.clip(1.0 - distances[offset],0.0,1.0)

                valid = ((indices[offset] != item_index) & (similarities > 0))

                selected = indices[offset][valid][:neighbors]
                selected_values = similarities[valid][:neighbors]

                if len(selected):
                    row_parts.append(np.full(len(selected), item_index, dtype=np.int32))

                    column_parts.append(selected.astype(np.int32))
                    value_parts.append(selected_values.astype(np.float32))

            LOGGER.info("ItemKNN: обработано %s/%s игр",stop, n_items)

        if row_parts:
            rows = np.concatenate(row_parts)
            columns = np.concatenate(column_parts)
            values = np.concatenate(value_parts)

            self.similarity = csr_matrix((values, (rows, columns)), shape=(n_items, n_items), dtype=np.float32)
        else:
            self.similarity = csr_matrix((n_items, n_items), dtype=np.float32)

        return self

    def score(self, user_index: int) -> np.ndarray:
        return (self.matrix.getrow(user_index) @ self.similarity).toarray().ravel()


class ALSModel:
    """ALS для неявной положительной обратной связи."""

    def fit(self, matrix: csr_matrix, factors: int, iterations: int, regularization: float, alpha: float, seed: int, threads: int):
        self.model = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            alpha=alpha,
            random_state=seed,
            use_gpu=False,
            num_threads=threads,
        )

        # Для implicit 0.7.x матрица имеет ориентацию: пользователи × игры.
        self.model.fit(matrix.astype(np.float32), show_progress=True)

        return self

    def score(self, user_index: int) -> np.ndarray:
        return (self.model.item_factors @ self.model.user_factors[user_index])


# Ранжирование и гибрид


def rank_scores(scores: np.ndarray, popularity: np.ndarray, seen: set[int], count: int) -> np.ndarray:
    """
    Ранжирует оценки и исключает известные игры. При равных оценках:
        1. выше популярность;
        2. меньше внутренний индекс.
    Это обеспечивает воспроизводимый порядок.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    allowed = np.isfinite(scores)

    if seen:
        excluded = np.fromiter(seen, dtype=np.int64)
        allowed[excluded] = False

    candidates = np.flatnonzero(allowed)

    if len(candidates) == 0:
        return np.empty(0, dtype=np.int64)

    order = np.lexsort((candidates, -popularity[candidates], -scores[candidates]))

    return candidates[order[:count]]

def combine_rrf(component_lists: list[np.ndarray], weights: np.ndarray, popularity: np.ndarray, top_k: int, rrf_constant: float) -> np.ndarray:
    """Вместо сложения несопоставимых сырых оценок объединяются позиции игр в списках моделей. Чем выше позиция игры в компоненте, тем больше её вклад."""
    item_parts = []
    contribution_parts = []

    for ranked_items, weight in zip(component_lists, weights):
        if weight <= 0 or len(ranked_items) == 0:
            continue

        ranks = np.arange(1, len(ranked_items) + 1, dtype=np.float64)

        item_parts.append(ranked_items)

        contribution_parts.append(weight / (rrf_constant + ranks))

    if not item_parts:
        return np.empty(0, dtype=np.int64)

    all_items = np.concatenate(item_parts)
    all_contributions = np.concatenate(contribution_parts)

    unique_items, inverse = np.unique(all_items, return_inverse=True)

    total_scores = np.bincount(inverse, weights=all_contributions, minlength=len(unique_items))

    order = np.lexsort((unique_items, -popularity[unique_items], -total_scores))

    return unique_items[order[:top_k]]

def build_rank_cache(models: dict[str, object], users: list[int], seen: dict[int, set[int]], popularity: np.ndarray, pool_size: int, phase: str) -> dict[int, list[np.ndarray]]:
    """Сохраняет списки кандидатов компонентов. Это позволяет перебирать веса гибрида без повторного запуска всех моделей для каждого набора весов."""
    cache = {}

    for number, user_index in enumerate(users, start=1):
        known = seen.get(user_index, set())
        component_lists = []

        for model_name in MODEL_NAMES:
            scores = models[model_name].score(user_index)

            ranked = rank_scores(scores=scores, popularity=popularity, seen=known, count=pool_size)
            component_lists.append(ranked)

        cache[user_index] = component_lists
        if number % 100 == 0 or number == len(users):
            LOGGER.info("%s: рекомендации для %s/%s пользователей",phase, number, len(users),)

    return cache

def weight_grid(parts: int):
    """Генерирует все комбинации четырёх неотрицательных весов."""
    for values in itertools.product(range(parts + 1), repeat=4):
        if sum(values) == parts:
            yield np.asarray(values, dtype=np.float64) / parts


# Часть - 7 Метрики


def user_metrics(recommended: np.ndarray, relevant: set[int], top_k: int) -> tuple[float, float, float]:
    """Precision, Recall и NDCG для одного пользователя."""
    hit_flags = np.array([int(item) in relevant for item in recommended[:top_k]], dtype=np.float64)

    hits = float(hit_flags.sum())

    precision = hits / top_k
    recall = hits / len(relevant) if relevant else 0.0

    discounts = np.log2(np.arange(len(hit_flags), dtype=np.float64) + 2.0)
    dcg = (float(np.sum(hit_flags / discounts)) if len(hit_flags) else 0.0)
    ideal_count = min(top_k, len(relevant))
    idcg = sum(1.0 / math.log2(position + 2) for position in range(ideal_count))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return precision, recall, ndcg

def tune_hybrid(cache: dict[int, list[np.ndarray]], targets: dict[int, set[int]], popularity: np.ndarray, top_k: int, grid_parts: int, rrf_constant: float) -> tuple[np.ndarray, pd.DataFrame]:
    """Выбирает веса по средней validation NDCG@K."""
    rows = []
    best_weights = None
    best_ndcg = -1.0

    for weights in weight_grid(grid_parts):
        ndcg_values = []

        for user_index, component_lists in cache.items():
            ranked = combine_rrf(component_lists, weights, popularity, top_k, rrf_constant)

            _, _, ndcg = user_metrics(ranked, targets[user_index], top_k)
            ndcg_values.append(ndcg)

        mean_ndcg = float(np.mean(ndcg_values))

        row = {
            "w_popularity": float(weights[0]),
            "w_tfidf": float(weights[1]),
            "w_itemknn": float(weights[2]),
            "w_als": float(weights[3]),
            f"validation_ndcg@{top_k}": mean_ndcg,
        }

        rows.append(row)

        # При равенстве сохраняется первая комбинация: порядок перебора детерминирован.
        if mean_ndcg > best_ndcg + 1e-12:
            best_ndcg = mean_ndcg
            best_weights = weights.copy()

    if best_weights is None:
        raise RuntimeError("Не удалось подобрать веса гибрида.")

    return best_weights, pd.DataFrame(rows)


def evaluate_test(
        cache: dict[int, list[np.ndarray]],
        targets: dict[int, set[int]],
        popularity: np.ndarray,
        weights: np.ndarray,
        top_k: int,
        rrf_constant: float,
        catalog: pd.DataFrame,
        user_ids: list[str],
        train_counts: np.ndarray,
        recommendation_users_limit: int,
):
    """Сравнивает компоненты и гибрид на одинаковой тестовой выборке."""
    metric_rows = []
    per_user_rows = []
    recommendation_rows = []

    coverage_sets = {model: set() for model in (*MODEL_NAMES, "Hybrid")}

    for user_number, (user_index, lists) in enumerate(cache.items()):
        ranked_by_model = {model_name: lists[index][:top_k] for index, model_name in enumerate(MODEL_NAMES)}

        ranked_by_model["Hybrid"] = combine_rrf(lists, weights, popularity, top_k, rrf_constant)

        for model_name, ranked in ranked_by_model.items():
            precision, recall, ndcg = user_metrics(ranked, targets[user_index], top_k)

            coverage_sets[model_name].update(int(item) for item in ranked)

            history_size = int(train_counts[user_index])

            if history_size < 10:
                segment = "less_than_10"
            elif history_size < 30:
                segment = "10_to_29"
            else:
                segment = "30_or_more"

            per_user_rows.append({
                "model": model_name,
                "user_id": user_ids[user_index],
                "train_interactions": history_size,
                "activity_segment": segment,
                f"precision@{top_k}": precision,
                f"recall@{top_k}": recall,
                f"ndcg@{top_k}": ndcg,
                "relevant_count": len(targets[user_index]),
                "recommendation_count": len(ranked),
            })

            if user_number < recommendation_users_limit:
                for rank, item_index in enumerate(ranked, start=1):
                    item_index = int(item_index)
                    game = catalog.iloc[item_index]

                    recommendation_rows.append({
                        "model": model_name,
                        "user_id": user_ids[user_index],
                        "rank": rank,
                        "app_id": game["app_id"],
                        "title": game["title"],
                        "is_relevant": (item_index in targets[user_index]),
                    })

    per_user = pd.DataFrame(per_user_rows)

    metric_columns = [
        f"precision@{top_k}",
        f"recall@{top_k}",
        f"ndcg@{top_k}",
    ]

    for model_name in (*MODEL_NAMES, "Hybrid"):
        selected = per_user[per_user["model"] == model_name]

        row = {
            "model": model_name,
            **{
                column: float(selected[column].mean())
                for column in metric_columns
            },
            f"coverage@{top_k}": (len(coverage_sets[model_name]) / len(catalog)),
            "evaluated_users": int(len(selected)),
        }

        metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)

    segments = (per_user.groupby(["model", "activity_segment"], as_index=False).agg
        (
            users=("user_id", "size"),
            precision=(f"precision@{top_k}", "mean"),
            recall=(f"recall@{top_k}", "mean"),
            ndcg=(f"ndcg@{top_k}", "mean"),
        )
    )

    recommendations = pd.DataFrame(recommendation_rows)

    return metrics, per_user, segments, recommendations


# Часть - 8 Новый пользователь


def recommend_new_user(preferences: str, content_model: ContentModel, popularity: np.ndarray, catalog: pd.DataFrame, top_k: int, pool_size: int, rrf_constant: float) -> pd.DataFrame:
    """Рекомендации без пользовательской истории."""
    popularity_rank = rank_scores(popularity, popularity, set(), pool_size)

    text_scores = content_model.score_text(preferences)

    if not preferences.strip() or not np.any(text_scores > 0):
        ranked = popularity_rank[:top_k]
        method = "Popularity: no recognized preferences"
    else:
        content_rank = rank_scores(text_scores, popularity, set(), pool_size)
        ranked = combine_rrf([popularity_rank, content_rank], np.array([0.30, 0.70]), popularity, top_k, rrf_constant)
        method = "TF-IDF + Popularity"

    result = catalog.iloc[ranked][["app_id", "title"]].copy()
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result["method"] = method

    return result


# Часть - 9 Итоговый вывод

def make_conclusion(metrics: pd.DataFrame, top_k: int, weights: np.ndarray, summary: dict,) -> str:
    """Создаёт вывод без заранее заданного победителя."""
    ndcg_column = f"ndcg@{top_k}"
    best_value = float(metrics[ndcg_column].max())
    winners = metrics[np.isclose( metrics[ndcg_column], best_value, rtol=0.0, atol=1e-12)]["model"].tolist()

    baseline = metrics.loc[metrics["model"] == "Popularity"].iloc[0]
    baseline_ndcg = float(baseline[ndcg_column])
    difference = best_value - baseline_ndcg

    if baseline_ndcg > 0:
        relative = (f"{100.0 * difference / baseline_ndcg:.2f}%")
    else:
        relative = ("не рассчитывается, поскольку NDCG базовой модели равна нулю")

    lines = [
        "ИТОГОВЫЙ ВЫВОД",
        "",
        (
            "В эксперименте использованы recommendations.csv, "
            "games.csv и users.csv."
        ),
        (
            f"Обучающая матрица: {summary['train_users']} пользователей, "
            f"{summary['train_items']} игр и "
            f"{summary['train_interactions']} положительных взаимодействий."
        ),
        (
            f"Валидационная оценка: "
            f"{summary['validation_evaluated_users']} пользователей."
        ),
        (
            f"Тестовая оценка: "
            f"{summary['test_evaluated_users']} пользователей."
        ),
        (
            "Наибольшая средняя тестовая NDCG"
            f"@{top_k}: {best_value:.6f}."
        ),
        "Модели с этим результатом: " + ", ".join(winners) + ".",
        (
            "Абсолютное отличие лучшего результата от Popularity: "
            f"{difference:.6f}."
        ),
        f"Относительное изменение: {relative}.",
        "",
        "ПОКАЗАТЕЛИ МОДЕЛЕЙ",
        ]

    for row in metrics.to_dict("records"):
        lines.append(
            f"{row['model']}: "
            f"Precision@{top_k}={row[f'precision@{top_k}']:.6f}; "
            f"Recall@{top_k}={row[f'recall@{top_k}']:.6f}; "
            f"NDCG@{top_k}={row[f'ndcg@{top_k}']:.6f}; "
            f"Coverage@{top_k}={row[f'coverage@{top_k}']:.6f}."
        )

    lines.extend([
        "",
        (
                "Выбранные на validation веса гибрида "
                "(Popularity, TF-IDF, ItemKNN, ALS): "
                + ", ".join(f"{value:.2f}" for value in weights)
                + "."
        ),
        "",
    ])

    return "\n".join(lines)

# Часть - 10 Аргументы для командной строки

def parse_arguments():
    parser = argparse.ArgumentParser(description="Гибридная рекомендательная система Steam")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-rows", type=int, default=0, help="0 — весь recommendations.csv; иначе первые N строк.")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--val-end", default=None)
    parser.add_argument("--min-user", type=int, default=5)
    parser.add_argument("--min-item", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=200)
    parser.add_argument("--max-eval-users", type=int, default=500, help="0 — все подходящие пользователи каждого периода.")
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument("--knn-neighbors", type=int, default=50)
    parser.add_argument("--knn-batch", type=int, default=128)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--als-factors", type=int, default=64)
    parser.add_argument("--als-iterations", type=int, default=15)
    parser.add_argument("--als-regularization", type=float, default=0.1)
    parser.add_argument("--als-alpha", type=float, default=20.0)
    parser.add_argument("--weight-parts", type=int, default=4, help="Шаг сетки весов равен 1 / weight-parts.")
    parser.add_argument("--rrf-constant", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-recommendation-users", type=int, default=20)
    parser.add_argument("--preferences", default="strategy adventure fantasy")

    args = parser.parse_args()

    positive_arguments = ["chunksize", "min_user", "min_item", "top_k", "pool_size", "max_features", "knn_neighbors", "knn_batch", "als_factors", "als_iterations", "weight_parts"]

    for name in positive_arguments:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} должен быть > 0.")

    if args.pool_size < args.top_k:
        parser.error("--pool-size должен быть не меньше --top-k.")

    if args.max_rows < 0 or args.max_eval_users < 0:
        parser.error("--max-rows и --max-eval-users должны быть >= 0.")

    if args.save_recommendation_users < 0:
        parser.error("--save-recommendation-users должен быть >= 0.")

    if args.jobs < 1:
        parser.error("--jobs должен быть >= 1.")

    if (args.als_regularization <= 0 or args.als_alpha <= 0 or args.rrf_constant <= 0):
        parser.error("Регуляризация ALS, alpha и rrf-constant должны быть > 0.")

    return args


# Часть - 11 Основной работа

def main():
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recommendations_path = data_dir / "recommendations.csv"
    games_path = data_dir / "games.csv"
    users_path = data_dir / "users.csv"

    LOGGER.info("Чтение recommendations.csv")
    events, recommendations_audit = load_recommendations(recommendations_path, args.max_rows)
    LOGGER.info("Чтение users.csv")
    users, users_audit = load_users(users_path, set(events["user_id"].unique()), args.chunksize)
    LOGGER.info("Чтение games.csv")
    games = load_games(games_path)
    before_user_join = len(events)

    # products и reviews сохраняются в отдельном users. Здесь проверяется принадлежность пользователей таблице users.csv.
    events = events.merge(users[["user_id"]], on="user_id", how="inner", validate="many_to_one")
    removed_unknown_users = before_user_join - len(events)
    before_game_filter = len(events)
    events = events[events["app_id"].isin(games["app_id"])].copy()
    removed_unknown_games = before_game_filter - len(events)
    if events.empty:
        raise ValueError("После согласования трёх CSV взаимодействий не осталось.")

    train_end, val_end = choose_boundaries(events, args.train_end, args.val_end)

    LOGGER.info("TRAIN < %s; VALIDATION < %s; далее TEST",train_end, val_end)

    train = events[events["date"].lt(train_end) & events["positive"]].copy()
    train_before_filter = len(train)
    train = k_core_filter(train, args.min_user, args.min_item)

    (matrix, catalog, user_ids, user_to_index, item_to_index) = build_training_matrix(train, games)

    if matrix.shape[1] <= args.top_k:
        LOGGER.warning("В каталоге мало игр относительно top-k. Некоторые списки могут быть короче K.")

    LOGGER.info("Матрица TRAIN: %s пользователей × %s игр; %s взаимодействий",matrix.shape[0], matrix.shape[1], matrix.nnz)

    seen_validation = build_seen(events, train_end, user_to_index, item_to_index)
    seen_test = build_seen(events, val_end, user_to_index, item_to_index)
    validation_targets, validation_audit = build_targets(events, train_end, val_end, user_to_index, item_to_index, seen_validation)
    test_targets, test_audit = build_targets(events, val_end, None, user_to_index, item_to_index, seen_test)
    validation_users = sample_users(validation_targets, args.max_eval_users, args.seed)
    test_users = sample_users(test_targets, args.max_eval_users, args.seed + 1)



    # ----------------------Обучение----------------------------

    LOGGER.info("Обучение Popularity")
    popularity_model = PopularityModel().fit(matrix)
    LOGGER.info("Обучение TF-IDF")
    content_model = ContentModel().fit(matrix, catalog, args.max_features)

    LOGGER.info("Обучение ItemKNN")

    knn_model = ItemKNNModel().fit(matrix, args.knn_neighbors, args.knn_batch, args.jobs)

    LOGGER.info("Обучение ALS")

    als_model = ALSModel().fit(matrix, args.als_factors, args.als_iterations, args.als_regularization, args.als_alpha, args.seed, args.jobs)
    models = {"Popularity": popularity_model, "TF-IDF": content_model, "ItemKNN": knn_model, "ALS": als_model}
    popularity = popularity_model.values


    # -----------------------Валидация гибрида---------------------------------

    validation_cache = build_rank_cache(models, validation_users, seen_validation, popularity, args.pool_size,"VALIDATION")

    weights, validation_results = tune_hybrid(validation_cache, validation_targets, popularity, args.top_k, args.weight_parts, args.rrf_constant)

    LOGGER.info("Выбранные веса [Popularity, TF-IDF, ItemKNN, ALS]: %s",weights.tolist())

    del validation_cache # Освобождение списков валидации перед тестом.


    # ------------------------Тест--------------------------------

    test_cache = build_rank_cache(models, test_users, seen_test, popularity, args.pool_size,"TEST")
    train_counts = np.diff(matrix.indptr)
    metrics, per_user, segments, recommendations = evaluate_test(
        test_cache,
        test_targets,
        popularity,
        weights,
        args.top_k,
        args.rrf_constant,
        catalog,
        user_ids,
        train_counts,
        args.save_recommendation_users,
    )



    # -------------------------Профили на основе users.csv-------------------------------

    train_profiles = pd.DataFrame({"user_id": user_ids, "train_positive_interactions": train_counts})
    user_profiles = train_profiles.merge(users, on="user_id", how="left", validate="one_to_one")


    # ---------------------------Новый пользователь-----------------------------

    new_user_recommendations = recommend_new_user(
        args.preferences,
        content_model,
        popularity,
        catalog,
        args.top_k,
        args.pool_size,
        args.rrf_constant,
    )


    # -------------------------Отчёт-------------------------------

    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])

    summary = {
        "parameters": vars(args),
        "recommendations_audit": recommendations_audit,
        "users_audit": users_audit,
        "games_loaded": int(len(games)),
        "events_removed_missing_user": int(removed_unknown_users),
        "events_removed_missing_game": int(removed_unknown_games),
        "train_end_exclusive": train_end.isoformat(),
        "validation_end_exclusive": val_end.isoformat(),
        "train_rows_before_k_core": int(train_before_filter),
        "train_users": int(matrix.shape[0]),
        "train_items": int(matrix.shape[1]),
        "train_interactions": int(matrix.nnz),
        "matrix_density": float(density),
        "matrix_sparsity": float(1.0 - density),
        "validation_audit": validation_audit,
        "test_audit": test_audit,
        "validation_evaluated_users": int(len(validation_users)),
        "test_evaluated_users": int(len(test_users)),
        "hybrid_component_order": list(MODEL_NAMES),
        "hybrid_weights": weights.tolist(),
        "metadata_note": (
            "games.csv — доступный снимок метаданных; "
            "историческое состояние не восстановлено."
        ),
        "users_features_note": (
            "products и reviews используются для описания профилей, "
            "но не входят в исторические признаки моделей."
        ),
        "evaluation_note": (
            "Warm-start. Модели обучены на TRAIN; "
            "веса гибрида подобраны на validation; "
            "тест не используется для настройки."
        ),
    }

    conclusion = make_conclusion(metrics, args.top_k, weights, summary)


    # ---------------------------Сохранение-----------------------------

    tables = {
        "metrics.csv": metrics,
        "validation_weights.csv": validation_results,
        "per_user_metrics.csv": per_user,
        "segment_metrics.csv": segments,
        "recommendations.csv": recommendations,
        "user_profiles.csv": user_profiles,
        "catalog.csv": catalog,
        "new_user_recommendations.csv": new_user_recommendations,
    }

    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")

    save_json(output_dir / "experiment_summary.json", summary)

    with (output_dir / "conclusion.txt").open("w", encoding="utf-8") as file:
        file.write(conclusion)

    print("\nРЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА\n")
    print(metrics.to_string(index=False, float_format="%.6f"))

    print("\nРЕКОМЕНДАЦИИ ДЛЯ НОВОГО ПОЛЬЗОВАТЕЛЯ\n")
    print(new_user_recommendations.to_string(index=False))

    print("\nИТОГОВЫЙ ВЫВОД\n")
    print(conclusion)

    LOGGER.info("Выполнение завершено. Результаты: %s",output_dir.resolve())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.warning("Выполнение остановлено пользователем.")
        sys.exit(130)
    except Exception:
        # Полная трассировка помогает определить причину ошибки: отсутствующий столбец, несовместимую библиотеку, недостаток памяти и другие проблемы.
        LOGGER.exception("Не удалось выполнить эксперимент.")
        sys.exit(1)


