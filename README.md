# CalcPro — Calculadora científica, CAS, graficador y tutor universitario

Aplicación web local en Python (Streamlit + SymPy + NumPy + SciPy + Plotly) pensada para estudiantes de ingeniería.

> **Estado de verificación.** El código completo está escrito y todos los archivos compilan sin errores de sintaxis.
> Sin embargo, **no pudo ejecutarse donde fue generado** (no había SymPy, Streamlit ni Plotly instalados ni acceso a internet).
> Por eso **la primera ejecución en su computadora es la prueba real**: corra `python -m pytest -q` y reporte cualquier fallo
> (con el mensaje completo) para corregirlo.

## 1. Instalación en Windows

1. Instale **Python 3.10 o superior** desde <https://www.python.org/downloads/> y marque **“Add python.exe to PATH”**.
2. Descomprima la carpeta `calculadora_cientifica` (por ejemplo en `C:\CalcPro`).
3. Abra **PowerShell** en esa carpeta y ejecute:

```powershell
cd C:\CalcPro\calculadora_cientifica
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Pruebe el núcleo matemático:

```powershell
python -m pytest -q
```

5. Inicie la aplicación:

```powershell
streamlit run app.py
```

Se abrirá el navegador en `http://localhost:8501`.

## 2. Solución de problemas frecuentes

| Problema | Solución |
|---|---|
| `python` no se reconoce | Reinstale Python marcando “Add to PATH”, o use `py -m venv .venv`. |
| `Activate.ps1 cannot be loaded… scripts disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y repita la activación; o use `cmd` con `.venv\Scripts\activate.bat`. |
| `streamlit` no se reconoce | Active el entorno virtual y use `python -m streamlit run app.py`. |
| Error al instalar `scipy`/`numpy` | `pip install --upgrade pip` y use Python 3.10–3.13 de 64 bits. |
| `ModuleNotFoundError: core` o `ui` | Ejecute siempre desde la carpeta que contiene `app.py`. |
| Una integral “tarda demasiado” | Hay un límite de 40 s; pruebe la versión definida (da aproximación numérica) o simplifique. |
| Puerto 8501 ocupado | `streamlit run app.py --server.port 8502` |
| Fórmulas LaTeX no se ven | Actualice el navegador/Streamlit (`pip install -U streamlit`). |

## 3. Estructura

```
calculadora_cientifica/
├─ app.py                  Punto de entrada y navegación
├─ requirements.txt
├─ core/                   Matemática (sin interfaz)
│  ├─ parser.py            Análisis SEGURO (sin eval): lista blanca de caracteres y nombres
│  ├─ calculator.py        Evaluación exacta/numérica, complejos
│  ├─ algebra.py           Simplificar, factorizar, ecuaciones, sistemas, polinomios
│  ├─ linalg.py            Matrices, determinantes, valores propios, LU, RREF
│  ├─ calculus.py          Derivadas, ∇, div, rot, J, H, críticos, Lagrange, integrales, línea/superficie
│  ├─ coordinates.py       Cartesianas, polares, cilíndricas, esféricas, vectores, rectas y planos
│  ├─ numerics.py          Series, EDO, raíces, interpolación, estadística, probabilidad
│  └─ safety.py            Límite de tiempo para cálculos costosos
├─ tutor/                  Modo Estudiante: pasos REALES verificados
│  ├─ derivatives.py  integrals.py  limits.py  equations.py  multiple.py  vector.py
│  ├─ exercises.py         Ejercicios similares con respuesta calculada
│  └─ model.py             Niveles, errores frecuentes, teoría, verificación
├─ plotting/graphs.py      2D, polar, paramétrica, implícita, 3D, campos, regiones (Plotly)
├─ ui/                     Páginas Streamlit, tema claro/oscuro, historial, fórmulas, ayuda
└─ tests/test_core.py      Pruebas con operaciones reales
```

## 4. Uso de cada módulo

- **Calculadora**: teclado por categorías o escritura directa (Enter evalúa). Historial, favoritos ★, recuperar ↺, memoria M+/MR/MC, ANS, exportar TXT/CSV/LaTeX/JSON, copiar con el icono del bloque de resultado. Etiquetas EXACTO/APROXIMADO.
- **Cálculo**: derivadas (ordinaria, superior, parcial, implícita, direccional), ∇, ∇·, ∇×, Jacobiano, Hessiano, puntos críticos con clasificación, Lagrange, límites; integrales indefinida, definida, impropia, doble, triple (cartesianas, polares, cilíndricas, esféricas), de línea, de superficie, flujo, longitud de arco, área de superficie, volumen.
- **Coordenadas y vectores**: conversiones con grados/radianes, gráfica 3D del punto, producto escalar/vectorial/mixto, ángulos, proyecciones, rectas y planos.
- **Graficador**: 14 tipos; zoom, desplazamiento, rotación 3D, colores, puntos, varias funciones, f′ e ∫f superpuestas.
- **Álgebra y matrices**: simplificar/factorizar/fracciones parciales, ecuaciones y sistemas (Gauss-Jordan con pasos), polinomios, 17 operaciones matriciales.
- **Herramientas**: Taylor, sumatorias, convergencia, EDO (simbólica y Runge-Kutta), Newton/bisección/secante, interpolación, estadística, distribuciones.
- **Biblioteca**: ~70 fórmulas en 15 temas, con variables, condiciones, uso y ejemplo.

### Modo Estudiante vs. Profesional
Cámbielos en la barra lateral en cualquier momento. Estudiante muestra el procedimiento con 4 niveles (el predeterminado es *Universitario completo*), pasos expandibles, “¿por qué?”, método alternativo, errores frecuentes y ejercicios similares.

## 5. Honestidad matemática (cómo funciona el “paso a paso”)

- Los pasos **se calculan de verdad** (no son plantillas). Cada resultado se **verifica**: derivadas contra SymPy y diferencia central numérica; integrales derivando la antiderivada; integrales definidas y múltiples con cuadratura numérica independiente; límites con límites laterales; ecuaciones por sustitución.
- Si ningún método didáctico reproduce el resultado, el paso lo declara (“resuelto con el algoritmo general de SymPy, sin desglose”). Nunca se inventa un procedimiento.
- Si una integral no tiene antiderivada elemental se da una aproximación **numérica** marcada como tal.

## 6. Convenciones
- `log(x)` = base 10 · `ln(x)` = natural · `log(x, b)` = base b.
- Esféricas (ρ, θ, φ): θ azimutal ∈ [0, 2π), φ polar desde +z ∈ [0, π]. (En física suelen ir intercambiados.)

## 7. Limitaciones conocidas
- Streamlit no permite atajos de teclado globales personalizados: se usa Enter, Tab y los botones.
- El tiempo límite de cálculo no puede matar el hilo: la interfaz responde, pero la CPU puede seguir ocupada un rato.
- Los teoremas de Green/Stokes/divergencia están en la biblioteca de fórmulas, pero no hay un módulo que los aplique automáticamente.
- Demostraciones formales no se generan; el nivel completo incluye fundamento teórico y condiciones de aplicación.
- Cálculo simbólico complejo (integrales difíciles, EDO no estándar) depende de SymPy y puede fallar o ser lento.

## 8. Recomendaciones para ampliar
Exportar solución a PDF/LaTeX; entrada con editor matemático (MathLive); módulo de Green/Stokes; Laplace y Fourier; persistencia del historial en disco; empaquetar como `.exe` con PyInstaller; pruebas con `hypothesis`.
