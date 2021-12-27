# Seminario IPD477

En este repositorio se encontrará todo el software para la replicación de los métodos desarrollados en Torres [1]. 

Para rerpoducir los resultados obtenidos en este trabajo se debe tener instalado Python (>=3.8), en particular algún toolkit que tenga la mayoría de las dependencias instaladas como Anaconda. Luego ejecutar en el terminal:

```
pip install -r requirements.txt
```
esto instalará las bibliotecas referenciadas en los scripts desarrollados.

Es necesario contar con la [base de datos] abierta utilizada para este trabajo [2]. Por defecto el extractor de características buscará una directorio arriba de este repositorio a la carpeta principal de la base de datos (../N2001), se recomienda mantener esta estructura. Luego para la extracción de características ejecutar *extract_features.py*. Para reducción de dimensionalidad y clasificación referirse al notebook *ulda_reduction.ipynb*.

En la carpeta **output** están los resultados de extracción utilizados en este trabajo y pueden ser utilizados directamente en el notebook *ulda_reduction.ipynb*.

## Referencias

[1] J. R. Torres-Castillo, C. O. Lopez-López, and M. A. Padilla-Castañeda, “Neuromuscular disorders detection through time-frequency analysis and classification of multi-muscular EMG signals using Hilbert-Huang transform”. Biomedical Signal Processing and Control, vol. 71, p. 103037, 2022.

[2] Nikolic M. "Detailed Analysis of Clinical Electromyography Signals EMG Decomposition, Findings and Firing Pattern Analysis in Controls and Patients with Myopathy and Amytrophic Lateral Sclerosis". PhD Thesis, Faculty of Health Science, University of Copenhagen, 2001.

[base de datos]: https://drive.google.com/drive/folders/1J1spfC8-SlO0QbL_dggUYj396gwWm0IY?usp=sharing