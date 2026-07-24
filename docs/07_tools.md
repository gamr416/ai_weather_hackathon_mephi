# Инструменты для погодных данных

Погодные поля похожи на изображения (CV), но с важными отличиями:

- геометрия / системы координат;
- много физических каналов (не RGB);
- единицы измерения и 3D-структура (уровни давления).

Поэтому нужны специальные форматы, метаданные и библиотеки.  
Сырой пост: [`more_info.md`](more_info.md). Для хакатона рекомендуют **xarray + Zarr** (WeatherBench 2 / ARCO ERA5).

---

## Must-have для этого проекта

| Библиотека | Зачем |
|------------|--------|
| **[xarray](https://docs.xarray.dev/)** | Тензоры с координатами (lat/lon/time/level), как numpy + метаданные |
| **[zarr](https://zarr.dev/)** | Чанковое облачное/локальное хранение (WB2 ERA5) |
| **[dask](https://www.dask.org/)** | Ленивая подгрузка больших массивов |
| **[gcsfs](https://gcsfs.readthedocs.io/)** | Чтение `gs://weatherbench2/...` |
| **[cfgrib](https://github.com/ecmwf/cfgrib) / eccodes** | Если качаете GRIB с CDS |
| **[netCDF4](https://unidata.github.io/netcdf4-python/)** | Классический NetCDF |

Уже стоят в `.venv` (кроме `gcsfs` — при необходимости: `pip install gcsfs`).

---

## Визуализация и геометрия

| Библиотека | Зачем |
|------------|--------|
| **[Cartopy](https://scitools.org.uk/cartopy/)** | Карты поверх matplotlib (проекции, берега) |
| **[matplotlib](https://matplotlib.org/)** | Базовые графики полей / спектров |
| **[Shapely](https://shapely.readthedocs.io/)** | Точки/линии/полигоны, геометрия |
| **[GeoPandas](https://geopandas.org/)** | Таблицы + геометрия (реже нужно) |
| **[rioxarray](https://corteva.github.io/rioxarray/) / [pyproj](https://pyproj4.github.io/pyproj/)** | CRS / проекции для растеров |

---

## Физика / диагностика (опционально)

| Библиотека | Зачем |
|------------|--------|
| **[MetPy](https://unidata.github.io/MetPy/)** | Термодинамика, сдвиг ветра, CAPE и т.п. |
| **[pvlib](https://pvlib-python.readthedocs.io/)** | Солнечная геометрия / радиация |

---

## «Тяжёлая артиллерия»

| Инструмент | Зачем |
|------------|--------|
| **[GDAL](https://gdal.org/)** (`osgeo`) | Быстрые warp/grid/конвертации больших растеров |

Для WB2 Zarr обычно достаточно xarray-стека без GDAL.

---

## Связь с ТЗ

Данные хакатона: §2.4 в [task.md](task.md) — Zarr WeatherBench 2, 28 каналов, 6h, 0.25°/0.5°.
