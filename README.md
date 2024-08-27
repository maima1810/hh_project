Здесь я проверил, можно ли подобрать подходящие загруженному резюме пользователя вакансии с hh.ru.
В качестве датасета использовалась выгрузка вакансий в сфере анализа данных.
Использовал следующие методы:
- парсинг вакансий в формате json с API hh.ru и загрузка их в датафрейм Pandas
- преобразование полученных данных, разведочный анализ датасета
- кластеризация вакансий методами К-средних и DBSCAN
- латентное распределение Дирихле (LDA)
- предобработка, векторизация текста резюме, по косинусной мере поиск ближайших вакансий
