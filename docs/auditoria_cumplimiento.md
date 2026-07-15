# Auditoría final de cumplimiento

| Requisito | Estado | Evidencia |
|---|---|---|
| Proyecto desarrollado en meta/proyecto_bien | Cumplido | Carpeta raíz del proyecto final |
| Dataset heart.csv incluido | Cumplido | data/heart.csv |
| Notebook único con flujo principal | Cumplido | Proyecto_GA_MLP_HeartDisease.ipynb |
| Carga del dataset | Cumplido | Sección 2 del notebook |
| Análisis exploratorio | Cumplido | results/eda_summary.json, tables/*.csv, figures/*.png |
| Preprocesamiento | Cumplido | ManualPreprocessor en el notebook |
| Modelo base MLP | Cumplido | Capas 64-32-16-1 con dropout |
| Algoritmo Genético | Cumplido | 50 generaciones, 20 individuos, cruce 0.8, mutación 0.1 |
| Modelo híbrido GA + MLP | Cumplido | results/best_ga_params.json y metrics.json |
| Entrenamiento y evaluación | Cumplido | results/metrics.json |
| Comparación de resultados | Cumplido | tables/tabla_4_comparacion_modelos.csv |
| Tablas obligatorias | Cumplido | tables/tabla_1 a tabla_4 |
| Gráficos | Cumplido | figures/*.png |
| Conclusiones | Cumplido | docs/informe_investigacion_formativa.docx |
| Informe formativo | Cumplido | docs/informe_investigacion_formativa.docx |
| Artículo IEEE | Cumplido | docs/articulo_ieee.docx |
| README y requirements | Cumplido | README.md, requirements.txt |
| Ejecución sin intervención manual | Cumplido | Notebook probado mediante ejecución secuencial |

## Resultados verificados

- F1 modelo base: 0.8190
- F1 modelo híbrido: 0.8505
- Mejor fitness GA en validación: 0.8495
- Figuras generadas: 9
- Tablas generadas: 9
