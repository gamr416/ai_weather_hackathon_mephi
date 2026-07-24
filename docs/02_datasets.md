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

**[WeatherBench 2](https://arxiv.org/abs/2308.15560)** ([GitHub](https://github.com/google-research/weatherbench2), [данные в GCS](https://console.cloud.google.com/storage/browser/weatherbench2)):

- готовые ERA5 / IFS и прогнозы DL-моделей;
- пайплайны и метрики;
- сравнение [детерминированных](https://sites.research.google.com/gr/weatherbench/deterministic-scores/) и [вероятностных](https://sites.research.google.com/gr/weatherbench/probabilistic-scores/) моделей.

### Официальный источник хакатона

```
gs://weatherbench2/datasets/era5/
1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr
```

- сетка **0.25°** (721×1440), шаг **6 h**, есть `total_precipitation_6hr`;
- выбрать 8 surface + T/U/V/Z/Q на levels `[1000,925,850,700]`;
- train **2014–2019**, val **2020**, test **2021**;
- сетка **0.5°** — conservative remapping → 360×720 (см. [task.md](task.md) §2.4–2.5).

Стек: `xarray` + `zarr` + `dask` + `gcsfs` — [07_tools.md](07_tools.md).

---

## Связь со сжатием

**CRA5 / VAEformer** — baseline non-inferiority (не цель по ~300×). В хакатоне целевой CR **×32–×64** при малом \(N\).
