import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. DEFINICIÓN DE LAS SEÑALES EN EL DOMINIO DEL TIEMPO
# =============================================================================
fs = 1000.0                        # Frecuencia de muestreo (1000 muestras por segundo)
Ts = 1.0 / fs                      # Periodo de muestreo (0.001 segundos)
t = np.arange(-1.0, 1.0, Ts)       # Vector de tiempo de -1 s a 1 s (2000 puntos)
N = len(t)                         # Número total de muestras

# A) Pulso rectangular: vale 1 entre -0.1 s y +0.1 s (duración tau = 0.2 s), y 0 fuera
tau = 0.2
x_pulso = np.where(np.abs(t) <= tau / 2.0, 1.0, 0.0)

# B) Función escalón unitario u(t): vale 0 antes de t = 0, y vale 1 para t >= 0
x_escalon = np.where(t >= 0.0, 1.0, 0.0)

# C) Función senoidal pura: amplitud 1 y frecuencia f0 = 15 Hz
f0 = 15.0
x_seno = np.sin(2 * np.pi * f0 * t)

# =============================================================================
# 2. CÁLCULO DE LA TRANSFORMADA DE FOURIER (FFT)
# =============================================================================
# Vector de frecuencias en Hz centrado en 0 Hz
freqs = np.fft.fftshift(np.fft.fftfreq(N, Ts))

def calcular_espectro(senal, factor_escala):
    # ifftshift alinea t = 0 para que la fase sea exacta antes de aplicar np.fft.fft()
    X_compleja = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(senal))) * factor_escala
    magnitud = np.abs(X_compleja)
    fase = np.angle(X_compleja)
    # Limpiamos el ruido numérico en la fase donde la magnitud es casi cero
    fase[magnitud < (0.02 * np.max(magnitud))] = 0.0
    return magnitud, fase

# Calculamos magnitud y fase para las 3 señales elementales
mag_pulso, fase_pulso     = calcular_espectro(x_pulso, Ts)
mag_escalon, fase_escalon = calcular_espectro(x_escalon, 1.0 / N)
mag_seno, fase_seno       = calcular_espectro(x_seno, 1.0 / N)

# =============================================================================
# 3. VISUALIZACIÓN DE RESULTADOS (TIEMPO, MAGNITUD Y FASE)
# =============================================================================
fig1, axs = plt.subplots(3, 3, figsize=(13, 8))
fig1.suptitle("Análisis de Señales Elementales: Dominio del Tiempo vs. Frecuencia", fontsize=14, fontweight='bold')

datos_senales = [
    ("Pulso Rectangular (τ = 0.2 s)", x_pulso, mag_pulso, fase_pulso, (-40, 40)),
    ("Escalón Unitario u(t)", x_escalon, mag_escalon, fase_escalon, (-30, 30)),
    ("Señal Senoidal (f0 = 15 Hz)", x_seno, mag_seno, fase_seno, (-40, 40))
]

for fila, (nombre, x_t, mag, fase, lim_f) in enumerate(datos_senales):
    # Columna 0: Dominio del tiempo
    axs[fila, 0].plot(t, x_t, color='#1B365D', lw=1.6)
    axs[fila, 0].set_title(f"{nombre} - Tiempo")
    axs[fila, 0].set_xlabel("Tiempo t (s)")
    axs[fila, 0].set_ylabel("Amplitud")
    axs[fila, 0].set_xlim(-0.4, 0.4)
    axs[fila, 0].grid(True, linestyle='--', alpha=0.6)

    # Columna 1: Espectro de Magnitud
    axs[fila, 1].plot(freqs, mag, color='#008080', lw=1.6)
    axs[fila, 1].set_title("Espectro de Magnitud |X(f)|")
    axs[fila, 1].set_xlabel("Frecuencia f (Hz)")
    axs[fila, 1].set_ylabel("Magnitud")
    axs[fila, 1].set_xlim(lim_f)
    axs[fila, 1].grid(True, linestyle='--', alpha=0.6)

    # Columna 2: Espectro de Fase
    axs[fila, 2].plot(freqs, fase, color='#D9534F', lw=1.4)
    axs[fila, 2].set_title("Espectro de Fase ∠X(f)")
    axs[fila, 2].set_xlabel("Frecuencia f (Hz)")
    axs[fila, 2].set_ylabel("Fase (rad)")
    axs[fila, 2].set_xlim(lim_f)
    axs[fila, 2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("1_senales_elementales.png", dpi=200)

# =============================================================================
# 4. VERIFICACIÓN DE PROPIEDADES DE LA TRANSFORMADA DE FOURIER
# =============================================================================
fig2, axs2 = plt.subplots(3, 2, figsize=(12, 8.5))
fig2.suptitle("Verificación de Propiedades: Linealidad, Desplazamiento y Escalamiento", fontsize=14, fontweight='bold')

# --- A) Propiedad de Linealidad: a*x1(t) + b*x2(t) ---
x1 = np.cos(2 * np.pi * 12.0 * t)   # Coseno de 12 Hz
x2 = np.cos(2 * np.pi * 35.0 * t)   # Coseno de 35 Hz
y_suma = 1.5 * x1 + 0.8 * x2        # Señal combinada

mag_x1, _   = calcular_espectro(x1, 1.0 / N)
mag_x2, _   = calcular_espectro(x2, 1.0 / N)
mag_suma, _ = calcular_espectro(y_suma, 1.0 / N)

axs2[0, 0].plot(t, y_suma, color='#1B365D', lw=1.4)
axs2[0, 0].set_title("Linealidad (Tiempo): y(t) = 1.5·cos(2π·12t) + 0.8·cos(2π·35t)")
axs2[0, 0].set_xlim(-0.15, 0.15)
axs2[0, 0].grid(True, linestyle='--', alpha=0.6)

axs2[0, 1].plot(freqs, mag_suma, color='#008080', lw=2.0, label="FFT{1.5·x1 + 0.8·x2}")
axs2[0, 1].plot(freqs, 1.5 * mag_x1 + 0.8 * mag_x2, '--', color='#D9534F', lw=1.5, label="1.5·|X1(f)| + 0.8·|X2(f)|")
axs2[0, 1].set_title("Linealidad (Frecuencia): Picos exactos en ±12 Hz y ±35 Hz")
axs2[0, 1].set_xlim(-50, 50)
axs2[0, 1].legend()
axs2[0, 1].grid(True, linestyle='--', alpha=0.6)

# --- B) Propiedad de Desplazamiento en el Tiempo: x(t - t0) ---
t0 = 0.15  # Retardo de 0.15 segundos
x_desplazado = np.where(np.abs(t - t0) <= tau / 2.0, 1.0, 0.0)
mag_desp, fase_desp = calcular_espectro(x_desplazado, Ts)

axs2[1, 0].plot(freqs, mag_pulso, color='#1B365D', lw=2.0, label="Magnitud Original")
axs2[1, 0].plot(freqs, mag_desp, '--', color='#D9534F', lw=1.6, label="Magnitud Desplazada (t0 = 0.15s)")
axs2[1, 0].set_title("Desplazamiento: Magnitud Invariante |X(f)|")
axs2[1, 0].set_xlim(-25, 25)
axs2[1, 0].legend()
axs2[1, 0].grid(True, linestyle='--', alpha=0.6)

axs2[1, 1].plot(freqs, np.unwrap(fase_pulso), color='#1B365D', lw=1.6, label="Fase Original")
axs2[1, 1].plot(freqs, np.unwrap(fase_desp), color='#D9534F', lw=1.6, label="Fase con Retardo (-2π·f·t0)")
axs2[1, 1].set_title("Desplazamiento: Cambio Lineal en la Fase ∠X(f)")
axs2[1, 1].set_xlim(-10, 10)
axs2[1, 1].legend()
axs2[1, 1].grid(True, linestyle='--', alpha=0.6)

# --- C) Propiedad de Escalamiento: Pulso angosto vs Pulso ancho ---
x_angosto = np.where(np.abs(t) <= 0.05, 1.0, 0.0)  # Duración 0.10 s
x_ancho   = np.where(np.abs(t) <= 0.20, 1.0, 0.0)  # Duración 0.40 s
mag_angosto, _ = calcular_espectro(x_angosto, Ts)
mag_ancho, _   = calcular_espectro(x_ancho, Ts)

axs2[2, 0].plot(t, x_angosto, color='#D9534F', lw=1.8, label="Pulso angosto (0.10 s)")
axs2[2, 0].plot(t, x_ancho, '--', color='#1B365D', lw=1.6, label="Pulso ancho (0.40 s)")
axs2[2, 0].set_title("Escalamiento (Tiempo): Compresión vs. Expansión")
axs2[2, 0].set_xlim(-0.4, 0.4)
axs2[2, 0].legend()
axs2[2, 0].grid(True, linestyle='--', alpha=0.6)

axs2[2, 1].plot(freqs, mag_angosto / np.max(mag_angosto), color='#D9534F', lw=1.8, label="Espectro pulso angosto (Se expande)")
axs2[2, 1].plot(freqs, mag_ancho / np.max(mag_ancho), '--', color='#1B365D', lw=1.6, label="Espectro pulso ancho (Se comprime)")
axs2[2, 1].set_title("Escalamiento (Frecuencia): Relación Inversa Tiempo-Frecuencia")
axs2[2, 1].set_xlim(-35, 35)
axs2[2, 1].legend()
axs2[2, 1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("2_propiedades_fourier.png", dpi=200)
plt.show()