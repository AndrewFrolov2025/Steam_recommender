# Steam_recommender


**РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА КОМПЬЮТЕРНЫХ ИГР ДЛЯ ПЛАТФОРМЫ STEAM**


Результат работы кода формируется в каталоге **outputs**


**Структура каталога:**

metrics.csv - Итоговое сравнение пяти моделей


validation_weights.csv - Все проверенные веса гибрида


per_user_metrics.csv - Метрики отдельных пользователей


segment_metrics.csv - Качество по длине обучающей истории


recommendations.csv - Примеры тестовых рекомендаций


user_profiles.csv - Профили с данными из users.csv


catalog.csv - Каталог основного эксперимента


new_user_recommendations.csv - Подборка по текстовым предпочтениям


experiment_summary.json - Параметры, даты и статистика обработки


conclusion.txt - Вывод, сформированный по реальным результатам


Параметры для запуска кода


>>     --data-dir data `
>>     --max-rows 0 `
>>     --max-eval-users 200 `
>>     --als-iterations 5 `
>>     --min-user 2 `
>>     --min-item 2   










