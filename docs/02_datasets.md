# Датасеты

## Кто собирает данные

| Организация | Что делает |
|-------------|------------|
| **ECMWF** | Европейский центр среднесрочных прогнозов; ERA5, IFS |
| **WMO** | Всемирная метеоорганизация |
| **UKMO / MetOffice** | Метеослужба Великобритании |
| **NOAA** | США; GFS и др. |

Централизованный открытый склад европейских данных: **[CDS (Climate Data Store)](https://cds.climate.copernicus.eu/)** от Copernicus / ECMWF.

---

## Популярные датасеты CDS

### ERA5 (главный «ground truth» в DL-погоде)

Реанализ: численная модель + спутники, радары, станции, буи.  
Глобальная сетка **0.25°**, шаг **1 час**, с 1950-х.

| Датасет | Содержание |
|---------|------------|
| [ERA5 pressure levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels) | Атмосфера на 37 уровнях |
| [ERA5 single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) | Наземные параметры |
| [ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) | Суша, сетка **0.1°** |

### Другие CDS

| Датасет | Простыми словами |
|---------|------------------|
| **E-OBS** | Наблюдения T и осадков по Европе, 0.1°, сутки |
| **CERRA** | Европейский реанализ ~5.5 км (удобен для downscaling) |

---

## Прогнозы численных моделей (не реанализ)

| Датасет | Простыми словами |
|---------|------------------|
| **IFS HRES** | Оперативный прогноз ECMWF, ~0.1°, шаг 6 ч (SOTA среди NWP) |
| **GFS025** | Прогноз NOAA GFS, 0.25° |
| **CMIP5/6** | Климатические проекции |

---

## Удобные агрегаты для ML

**[WeatherBench2](https://arxiv.org/abs/2308.15560)** ([GitHub](https://github.com/google-research/weatherbench2), [данные в GCS](https://console.cloud.google.com/storage/browser/weatherbench2)):

- готовые ERA5 / IFS и прогнозы DL-моделей;
- пайплайны и метрики;
- сравнение [детерминированных](https://sites.research.google.com/gr/weatherbench/deterministic-scores/) и [вероятностных](https://sites.research.google.com/gr/weatherbench/probabilistic-scores/) моделей.

---

## Связь со сжатием

**CRA5** — сжатый ERA5 через VAEformer (~300–470×): из сотен TB → <1 TB при приемлемой ошибке. Это ближайший «продуктовый» аналог идеи Perceiver-VAE как кодека.
