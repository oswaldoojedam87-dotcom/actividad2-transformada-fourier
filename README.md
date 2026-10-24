# Actividad Formativa 2: Simulación y análisis de señales con la Transformada de Fourier

En este repositorio comparto el código en Python desarrollado para analizar señales en el dominio del tiempo y de la frecuencia aplicando la Transformada Rápida de Fourier (`np.fft.fft()`).

## Contenido del proyecto
* **`simulacion_fourier.py`**: Código principal comentado paso a paso donde se generan el pulso rectangular, la función escalón y la señal senoidal, se grafican sus espectros de magnitud y fase, y se comprueban las propiedades de linealidad, desplazamiento en el tiempo y escalamiento.
* **`1_senales_elementales.png` y `2_propiedades_fourier.png`**: Gráficas generadas por la simulación.

## Cómo ejecutar el código
Para correr el programa se necesita tener instalado Python junto con las librerías `numpy`, `scipy` y `matplotlib`:
```bash
pip install numpy scipy matplotlib
python simulacion_fourier.py