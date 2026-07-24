# Сырой дамп (архив)

Этот файл — исходный неразмеченный текст (описание Perceiver-VAE, погодные параметры, датасеты, конспект Aurora).

**Читать лучше структурированную базу:**

→ [README.md](README.md) (оглавление)

Ниже — оригинал без правок смысла.

---

Perceiver-VAE: A Universal Latent Codec for Weather Fields
Description
Perceiver-VAE - the idea is to leverage the Perceiver IO approach-successfully applied in the Aurora foundation weather model-to universally encode meteorological variables into a unified VAE latent space. The study centers on fine-tuning pretrained Perceiver-VAE models for new types of weather variables. The primary applications are compression of weather fields and building forecasting models directly on the universal VAE latents.

The Perceiver IO architecture.
TIMES (T-1T)
Perceiver-based Weather Model.
AIRI
TIMET+1
Tasks
Unify & tokenize data: dataloaders for reanalyses, surface fields, satellite swaths; encode (lat, lon, level, variable, time) with positional/spherical features; handle masks/missing values.
Train the codec: Perceiver-IO VAE with rate-distortion objective; add physics-aware and spectral regularizers; curriculum from coarse fine; mixed-precision + activation checkpointing.
Query-driven decoding & evaluation: design variable/level-aware query heads; support arbitrary output grids; evaluate RMSE/ACC/CRPS, power spectra.
Links
Perceiver IO: A General Architecture for Structured Inputs & Outputs (ICML 2022)
CRA5: Extreme Compression of ERAS for Portable Global Climate and Weather Research via an Efficient Variational Transformer
Aurora: A Foundation Model for the Earth System


Погодные параметры
Данный пост служит справочной информацией о физических параметрах, используемых в работе погодных моделей.

Погодные данные можно классифицировать на три типа:

Наземные
двумерные поля, представленные в окрестности поверхности Земли, они чаще всего используются в прикладных задачах, к ним можно отнести:
• Температура над 2 метрами от поверхности (T2M (https://codes.ecmwf.int/grib/param-db/167));
• Атм. давление на уровне моря (MSLP (https://codes.ecmwf.int/grib/param-db/151));
• Горизонтальные компоненты скорости ветра над 10 метрами от поверхности (U10 (https://codes.ecmwf.int/grib/param-db/165), V10 (https://codes.ecmwf.int/grib/param-db/166));
• Количество осадков за N часов (TP (https://codes.ecmwf.int/grib/param-db/228)) [в качестве N обычно смотрят на 1, 6, 12 или 24 часа];
(более редкие параметры)
• Температура поверхности моря (SST (https://codes.ecmwf.int/grib/param-db/34));
• Температура точки росы (D2M (https://codes.ecmwf.int/grib/param-db/3017));
• Полная (тотальная) облачность (TCC (https://codes.ecmwf.int/grib/param-db/164));
• Наземное атм. давление (SP (https://codes.ecmwf.int/grib/param-db/134));
• Общий столб воды (TCW/TCWP (https://codes.ecmwf.int/grib/param-db/136));
• Объем выпавшего снега (SF (https://codes.ecmwf.int/grib/param-db/144));
• Сток воды (RO (https://codes.ecmwf.int/grib/param-db/205));
• Горизонт. комп. скорости ветра на 100 метрах (U100 (https://codes.ecmwf.int/grib/param-db/228246), V100 (https://codes.ecmwf.int/grib/param-db/228247));
(радиационные параметры)
• Входящий коротковолновой радиационный поток на поверхность (SSRD (https://codes.ecmwf.int/grib/param-db/169));
• Входящий длинноволновой рад. (тепловой) поток на поверхность (STRD (https://codes.ecmwf.int/grib/param-db/175));
• Скрытый тепловой поток с поверхности (SLHF (https://codes.ecmwf.int/grib/param-db/147));
• Солнечная радиация на верхних слоях атмосферы (TISR (https://codes.ecmwf.int/grib/param-db/212));
• Длинноволновой рад. (тепловой) поток на верхних слоях атмосферы (TTR (https://codes.ecmwf.int/grib/param-db/179));
(параметры, связанные с почвой)
• Испарение/evapotranspiration за N часов (EVAP/E (https://codes.ecmwf.int/grib/param-db/182));
• Влажность почвы по слоям (SWVL1 (https://codes.ecmwf.int/grib/param-db/39)–SWVL4 (https://codes.ecmwf.int/grib/param-db/42));
• Температура почвы по слоям (STL1 (https://codes.ecmwf.int/grib/param-db/139)-STL4 (https://codes.ecmwf.int/grib/param-db/236));


Атмосферные
трёхмерные поля, представленные с учетом вертикальных (σ-) координат, распределенных по N атмосферным слоям. Исторически, атм. слои распределяют не по метрам, а по уровням давлений, измеряемых в гектопаскалях (гПа). 
В качестве примера распределния атм. слоев можно рассмотреть набор из 13 уровней, которые чаще всего используются в DL моделях погоды (от поверхности до верхних слоев атмосферы в гПа):
1000 → 925 → 850 → 700 → 600 → 500 → 400 → 300 → 250 → 200 → 150 → 100 → 50
Основные атмосферные параметры:
• Температура (T (https://codes.ecmwf.int/grib/param-db/130));
• Геопотенциал (Z (https://codes.ecmwf.int/grib/param-db/129));
• Удельная влажность (Q (https://codes.ecmwf.int/grib/param-db/133));
• Относительная влажность (R (https://codes.ecmwf.int/grib/param-db/157));
• Горизонтальные компоненты скорости ветра (U (https://codes.ecmwf.int/grib/param-db/131), V (https://codes.ecmwf.int/grib/param-db/132));
• Вертикальная компонента скорости ветра (W (https://codes.ecmwf.int/grib/param-db/135));


Статические (топографические)
двумерные поля, включающие дополнительную информацию о факторах, влияющих на изменение атмосферы. Эти поля (условно) неизменны во времени (их не требуется считать в процессе работы моделей).
• Координатная сетка (lat (https://codes.ecmwf.int/grib/param-db/250001)/lon (https://codes.ecmwf.int/grib/param-db/250002));
• Маска суши/моря (lsm (https://codes.ecmwf.int/grib/param-db/172));
• Карта типов почв (slt (https://codes.ecmwf.int/grib/param-db/43));
• Высота рельефа / наземная геоп. высота (orography/gh (https://codes.ecmwf.int/grib/param-db/156)/z);
• Стандартное отклонение субсеточного рельефа (sdor (https://codes.ecmwf.int/grib/param-db/160));
• Наклон субсеточного рельефа (slor (https://codes.ecmwf.int/grib/param-db/163));
• Анизотропия субсеточного рельефа (isor (https://codes.ecmwf.int/grib/param-db/161));
• Угол ориентации субсеточного рельефа (anor (https://codes.ecmwf.int/grib/param-db/162));
(параметры времени)
• Час в сутках (hod) (обычно отношение h/24);
• День в году (doy) (обычно отношение d/366);

Пример
Классическим набором погодных параметров можно считать следующие данные: 
[Z, T, Q, U, V] на 13 атм. слоях + [T2M, MSLP, U10, V10, TP], т.е. 70 погодных параметров.

В качестве классического набора топографических признаков можно выделить: [gh, lsm, slt].


Подробное описание каждого из погодных параметров можно найти на сайте документации ECMWF (https://codes.ecmwf.int/grib/param-db/).

#weather_data


Погодные датасеты
В мире существует множество организаций, собирающих/фильтрующих/складирующих погодные данные. К наиболее известным относятся:
• Европейский центр среднесрочных прогнозов погоды (ECMWF) (https://www.ecmwf.int/),
• Всемирная метеорологическая организация (WMO) (https://wmo.int/), 
• Метеорологическая служба Великобритании (UKMO/MetOffice) (https://www.metoffice.gov.uk/),
• Национальное управление океанических и атмосферных исследований США (NOAA) (https://www.noaa.gov/)
и ещё множество других организаций.

Для упрощения доступа и эффективной актуализации погодных данных ECMWF, Copernicus (https://www.copernicus.eu/) и другие европейские организации поддерживают и обновляют централизованное хранилище погодных датасетов — Climate Data Storage (CDS) (https://cds.climate.copernicus.eu/). Это открытая платформа, где каждый может выбрать необходимые датасеты прогнозов/наблюдений/реанализа различных моделей, настроить пространственные и временные диапазоны с интересующими параметрами и выгрузить эти данные себе локально!

Примеры популярных датасетов CDS
ERA5
Класс датасетов, представленных реанализом (прогнозы численных моделей, снабженные информацией со спутников/радаров/станций/морских буев). Данные реанализа распределены на глобальной сетке с шагом 0.25° по часовой временной сетке с 1950-х годов по настоящее время. Эти датасеты в ряде исследований принято считать за Ground Truth данные ввиду их покрытия количества ассимилируемых данных. Примеры конкретных датасетов:
• ERA5 hourly data on pressure levels (https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview) (реанализ погоды на 37 атмосферных слоях);
• ERA5 hourly data on single levels (https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview) (реанализ наземных погодных параметров);
• ERA5-Land hourly data (https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview) (детализированный реанализ наземных параметров на сетке 0.1° по суше);

E-OBS (https://cds.climate.copernicus.eu/datasets/insitu-gridded-observations-europe?tab=overview)
Датасет наблюдений по температурам и осадкам по территории Европы, интерполированный по сетке 0.1° с суточным шагом с 1950 года по настоящее время.

CERRA (https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-pressure-levels?tab=overview)
Датасет реанализа атмосферных погодных параметров по Европе с мезомасштабным разрешением 5.5км (полезен для тестирования моделей понижения масштаба).

Другие виды датасетов
IFS HRES (https://www.ecmwf.int/en/forecasts/datasets/set-i)
Результаты прогнозирования численной модели ECMWF IFS (SOTA среди численных моделей). Данные представлены на сетке 0.1° с 6-часовым шагом по наземным и атмосферным слоям.

GFS025 (https://rda.ucar.edu/datasets/d084001/)
Датасет на основе прогнозов численной модели GFS (NOAA).

CMIP5/6 (https://pcmdi.llnl.gov/CMIP6/)
Региональные датасеты климатических проекций.

Современные подходы к агрегации погодных данных
В 2022 году компания Google представила (https://sites.research.google/gr/weatherbench/) удобный облачный интерфейс для работы с погодными датасетами — как с классическими ERA5/IFS, так и с прогнозами DL моделей, актуальных на тот момент. На данный момент актуальной версией сервиса является WeatherBench2 (https://arxiv.org/pdf/2308.15560) (github (https://github.com/google-research/weatherbench2)), включающей помимо данных набор пайплайнов тестирования и метрик оценки качества моделей прогноза погоды (Web-интерфейс для сравнения качества детерминированных (https://sites.research.google/gr/weatherbench/deterministic-scores/) и вероятностных (https://sites.research.google/gr/weatherbench/probabilistic-scores/) моделей) . Сами данные расположены тут (https://console.cloud.google.com/storage/browser/weatherbench2).

#weather_data


Aurora: A Foundation Model for the Earth System (https://arxiv.org/pdf/2405.13063)
Класс фундаментальных моделей прогноза погоды от Microsoft, ориентированных на погоду, океан, качество воздуха и циклоны. Главные особенности работы:
• исследование механизмов дообучения модели на долгосрочных дистанциях прогнозирования посредством LoRA (https://arxiv.org/pdf/2106.09685) адаптеров;
• модификация блоков Encoder/Decoder с возможностью дообучения модели на новых классах погодных параметров;

⛅️ Погодные параметры
Для основной модели классический набор (https://t.me/weatherpapers/4): [Z, T, Q, U, V] × 13 слоев + наземные [T2M, U10, V10, MSLP] + статичные [lsm, slt, z]. Итого 69 погодных каналов + 3 стат. поля.

Для модели прогноза качества воздуха (Air Pollution (https://microsoft.github.io/aurora/models.html#aurora-0-4-air-pollution)) добавляются ещё атмосферные каналы [CO, NO, NO2, SO2, GO3] и 8 наземных (воздушных каналов) + 8 химических стат. полей.

Для океанологической модели (Wave (https://microsoft.github.io/aurora/models.html#aurora-0-25-wave)) к классическому набору добавляется 9 гидрологических параметров и 2 стат. поля.

Все погодные параметры приведены в таблице на рис. 2.

📐 Масштабы модели
Все рассматриваемые ниже модели работают с 6-часовым временным шагом (однако есть и 12-часовая модификация для Air Pollution).

По пространственному масштабу:
Aurora: 0.25°, 0.1°
Aurora Air Pollution: 0.4°
Aurora Wave: 0.25°

☀️ Датасеты (тут их много...)
(Из-за количества датасетов придется просто указать их названия и top-1/2/3 датасета по объему в обучающей выборке)
• ERA5 (top-1);
• GFS forecasts/T0 (top-2);
• HRES forecasts/analysis/T0/WAM (top-3);
• IFS ENS (+mean);
• GEFS;
• CMIP6;
• MERRA-2;
• CAMS (+reanalysis);

Подробности об объемах и разбиениях датасетов есть в статье.

⚛️ Архитектура
Архитектура Aurora (рис. 1) претерпела значительных изменений относительно FuXi (https://t.me/weatherpapers/19) и Pangu-Weather (https://t.me/weatherpapers/12) моделей (кроме 48 блоков SwinTransformer'а [уже  база]). Первой особенностью является замена Patch Embedding слоев в Encoder/Decoder частях на подход 3D Perceiver (https://arxiv.org/pdf/2107.14795). 

Набор входных данных представлен двумя последовательными временными снимками [t-6h,t] и подается на вход 3D Perceiver Encoder, который преобразует физ. поля в погодные эмбеддинги с помощью MLP и последовательности cross-attention блоков. Эти cross-attention блоки позволяют ослабить зависимость архитектуры от конкретных видов погодных параметров, их масштаба (от 0.4° до 0.1°) и числа атм. слоев (13-37) за счет учета эмбеддингов: [positional, scale, level, time]. Доп. каналы требуют лишь добавления отдельных MLP блоков перед Perceiver для учета новых погодных полей.
(часть описания Perceiver достаточно объемная, поэтому лучше почитать о ней в статье в п. B.1)

После получения погодных эмбеддингов из физ. полей идет UNet модель с тремя encoder/decoder ступенями, представленными в виде наборов классических Swin 3D Transformer блоков (Encoder: [6,10,8], Decoder: [8,10,6] — суммарно 48 блоков со всех ступеней). В ходе обучения эти блоки допускают модификации в виде LoRA адапторов на Attention/MLP части блоков Swin 3D Transformer.

На выходе погодные эмбеддинги проходят декодирование через 3D Perceiver Decoder, обеспечивающий восстановление погодных полей из эмбеддингов через cross-attention блоки. (fun fact: на этом этапе в роли Q-векторов выступают эмбеддинги, кодирующие давления атм. уровней)

📚 Процесс обучения
Обычный авторегрессионный пайплайн обучения с небольшими нововведениями:
• новые физ. корректные веса у функции потерь на основе взвешенной L1 (MAE);
• модели обучаются на разных датасетах и на разных масштабах, регулируемые стат. параметром scale и patch_size в архитектуре;
• на этапе обучения на долгосрочных дистанциях авторегрессии к модели присоединяют LoRA адаптеры для упрощения дообучения;
• модель допускает дообучение (без запуска обучения заново) на новых класссах погодных параметров. Для этого требуется только добавить новые MLP слои под каждый параметр в Perceiver Encoder и Decoder частях.

⚡️ Inference
Авторегрессия. Для запуска моделей требуется прокидывать доп. информацию о масштабах, количествах погодных параметров и атм. слоев.  

📈 Качество
Метрики на рис. 3.

📌 Доп. материалы
Репозиторий (https://github.com/microsoft/aurora) с весами и архитектурами моделей

Этапы работы алгоритма
📥 Сбор входных данных: Принимаются метеорологические карты за прошлые шаги времени (T-1, T).
⚙️ Кодирование признаков: Блок 3D Perceiver Encoder обрабатывает переменные и давление.
📦 Создание скрытого пространства: Данные упаковываются в трехмерный тензор Latent Atmospheric Input.
🔄 Пространственно-временной анализ: Модель 3D Swin Transformer UNet обрабатывает этот тензор.
🛠️ Адаптация весов: Модуль LoRA дообучает модель под конкретную задачу.
📊 Генерация скрытого прогноза: Формируется обновленный выходной тензор Latent Atmospheric Output.
🔓 Декодирование результатов: Блок 3D Perceiver Decoder переводит скрытые данные в физические переменные.
🌍 Вывод прогноза: Выдается готовая карта состояния атмосферы на шаг (T+1).
