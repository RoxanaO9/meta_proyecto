# Proyecto GA + MLP para clasificación de enfermedad cardíaca

Proyecto académico reconstruido dentro de `meta/proyecto_bien`.

## Ejecución

1. Instalar dependencias:

```bash
pip install -r requirements.txt
```

2. Abrir y ejecutar de principio a fin:

```text
Proyecto_GA_MLP_HeartDisease.ipynb
```

## Estructura

- `Proyecto_GA_MLP_HeartDisease.ipynb`: flujo principal completo.
- `data/heart.csv`: dataset usado.
- `docs/informe_investigacion_formativa.docx`: informe final.
- `docs/articulo_ieee.docx`: propuesta de artículo IEEE.
- `figures/`: gráficos generados.
- `tables/`: tablas obligatorias y auxiliares.
- `results/`: métricas, historial GA y parámetros óptimos.
- `models/`: modelos serializados y preprocesador.

## Resultado principal

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC AUC | Tiempo (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Modelo Base | 0.7935 | 0.7963 | 0.8431 | 0.8190 | 0.8758 | 1.0482 |
| Modelo Híbrido GA + MLP | 0.8261 | 0.8125 | 0.8922 | 0.8505 | 0.8934 | 0.5855 |

## Nota

El cronograma menciona 12 atributos + 1 objetivo, pero el archivo real contiene 11 variables predictoras + 1 objetivo. El proyecto usa el dataset real como fuente de verdad y documenta esa diferencia en la auditoría.
