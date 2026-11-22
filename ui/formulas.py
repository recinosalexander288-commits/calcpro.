"""Biblioteca de fórmulas organizada por temas.

Cada entrada: nombre, expresión (LaTeX), variables, condiciones, cuándo se usa y ejemplo.
"""
from __future__ import annotations


def F(name, latex, variables, conditions, use, example):
    return dict(name=name, latex=latex, variables=variables, conditions=conditions, use=use, example=example)


LIBRARY: dict[str, list[dict]] = {
    "Álgebra": [
        F("Fórmula general (cuadrática)", r"x=\frac{-b\pm\sqrt{b^{2}-4ac}}{2a}", "a, b, c: coeficientes de ax²+bx+c=0", "a ≠ 0", "Resolver ecuaciones de segundo grado.", "x²−5x+6=0 → x = (5 ± 1)/2 → x=3, x=2"),
        F("Binomio de Newton", r"(a+b)^{n}=\sum_{k=0}^{n}\binom{n}{k}a^{n-k}b^{k}", "n: entero ≥ 0; k: índice", "n ∈ ℕ", "Desarrollar potencias de binomios.", "(x+1)³ = x³+3x²+3x+1"),
        F("Diferencia de cuadrados", r"a^{2}-b^{2}=(a-b)(a+b)", "a, b: expresiones", "—", "Factorizar y racionalizar.", "x²−9=(x−3)(x+3)"),
        F("Propiedades de logaritmos", r"\log_b(xy)=\log_b x+\log_b y,\quad \log_b x^{n}=n\log_b x", "b: base, x,y: argumentos", "b>0, b≠1; x,y>0", "Simplificar y resolver ecuaciones logarítmicas.", "log₂8 = 3"),
        F("Cambio de base", r"\log_b x=\frac{\ln x}{\ln b}", "b: base", "b>0, b≠1; x>0", "Calcular logaritmos en cualquier base.", "log₅20 = ln20/ln5"),
    ],
    "Trigonometría": [
        F("Identidad pitagórica", r"\sin^{2}\theta+\cos^{2}\theta=1", "θ: ángulo", "—", "Simplificar expresiones y sustituciones trigonométricas.", "cos²θ = 1 − sen²θ"),
        F("Ángulo doble", r"\sin 2\theta=2\sin\theta\cos\theta,\quad \cos 2\theta=\cos^{2}\theta-\sin^{2}\theta", "θ: ángulo", "—", "Integrales de sen², cos²; simplificación.", "sen²θ = (1−cos2θ)/2"),
        F("Suma de ángulos", r"\sin(\alpha\pm\beta)=\sin\alpha\cos\beta\pm\cos\alpha\sin\beta", "α, β: ángulos", "—", "Evaluar ángulos notables.", "sen75° = sen(45°+30°)"),
        F("Ley de senos", r"\frac{a}{\sin A}=\frac{b}{\sin B}=\frac{c}{\sin C}", "a,b,c: lados; A,B,C: ángulos opuestos", "Triángulo", "Resolver triángulos con un lado y dos ángulos.", "—"),
        F("Ley de cosenos", r"c^{2}=a^{2}+b^{2}-2ab\cos C", "a,b,c lados; C ángulo opuesto a c", "Triángulo", "Resolver triángulos con dos lados y el ángulo comprendido.", "—"),
    ],
    "Geometría analítica": [
        F("Distancia entre puntos", r"d=\sqrt{(x_2-x_1)^{2}+(y_2-y_1)^{2}+(z_2-z_1)^{2}}", "P₁(x₁,y₁,z₁), P₂(x₂,y₂,z₂)", "—", "Longitudes y distancias.", "d((0,0,0),(1,2,2)) = 3"),
        F("Ecuación del plano", r"a(x-x_0)+b(y-y_0)+c(z-z_0)=0", "(a,b,c): normal; (x₀,y₀,z₀): punto del plano", "(a,b,c) ≠ 0", "Describir planos.", "Normal (1,2,3) y punto (0,0,0): x+2y+3z=0"),
        F("Distancia punto–plano", r"d=\frac{|ax_0+by_0+cz_0+d|}{\sqrt{a^{2}+b^{2}+c^{2}}}", "(x₀,y₀,z₀): punto; ax+by+cz+d=0: plano", "—", "Distancia mínima a un plano.", "—"),
        F("Ecuación paramétrica de la recta", r"\mathbf r(t)=\mathbf r_0+t\,\mathbf v", "r₀: punto; v: dirección; t∈ℝ", "v ≠ 0", "Describir rectas en el espacio.", "—"),
    ],
    "Vectores": [
        F("Producto escalar", r"\mathbf a\cdot\mathbf b=\sum a_ib_i=\|\mathbf a\|\|\mathbf b\|\cos\theta", "a, b: vectores; θ: ángulo entre ellos", "Misma dimensión", "Ángulos, proyecciones, trabajo.", "(1,2,3)·(4,5,6)=32"),
        F("Producto vectorial", r"\mathbf a\times\mathbf b=\begin{vmatrix}\mathbf i&\mathbf j&\mathbf k\\a_1&a_2&a_3\\b_1&b_2&b_3\end{vmatrix}", "a, b ∈ ℝ³", "Sólo en ℝ³", "Vector normal, áreas de paralelogramos, momento.", "i×j=k"),
        F("Ángulo entre vectores", r"\theta=\arccos\frac{\mathbf a\cdot\mathbf b}{\|\mathbf a\|\|\mathbf b\|}", "a, b: vectores", "a, b ≠ 0", "Ortogonalidad (θ=90°).", "—"),
        F("Proyección vectorial", r"\operatorname{proy}_{\mathbf b}\mathbf a=\frac{\mathbf a\cdot\mathbf b}{\|\mathbf b\|^{2}}\mathbf b", "a, b: vectores", "b ≠ 0", "Descomponer un vector.", "—"),
    ],
    "Matrices": [
        F("Determinante 2×2", r"\det\begin{pmatrix}a&b\\c&d\end{pmatrix}=ad-bc", "a, b, c, d: entradas", "Matriz 2×2", "Invertibilidad, áreas.", "det[[1,2],[3,4]] = −2"),
        F("Inversa 2×2", r"A^{-1}=\frac{1}{ad-bc}\begin{pmatrix}d&-b\\-c&a\end{pmatrix}", "A: matriz 2×2", "det A ≠ 0", "Resolver Ax=b.", "—"),
        F("Polinomio característico", r"\det(A-\lambda I)=0", "λ: valor propio", "A cuadrada", "Hallar valores propios.", "—"),
        F("Regla de Cramer", r"x_i=\frac{\det A_i}{\det A}", "A_i: A con la columna i reemplazada por b", "det A ≠ 0", "Sistemas pequeños.", "—"),
    ],
    "Límites": [
        F("Límite notable seno", r"\lim_{x\to0}\frac{\sin x}{x}=1", "x en radianes", "x → 0", "Resolver indeterminaciones trigonométricas.", "lim sen3x / x = 3"),
        F("Límite notable de e", r"\lim_{x\to\infty}\left(1+\frac1x\right)^{x}=e", "—", "x → ∞", "Formas 1^∞.", "lim (1+2/x)^x = e²"),
        F("Regla de L'Hôpital", r"\lim\frac{f(x)}{g(x)}=\lim\frac{f'(x)}{g'(x)}", "f, g derivables", "Forma 0/0 o ∞/∞; el límite derecho debe existir", "Resolver indeterminaciones.", "lim (eˣ−1)/x = 1"),
        F("Límite de exponencial menos 1", r"\lim_{x\to0}\frac{e^{x}-1}{x}=1", "—", "x → 0", "Indeterminaciones con eˣ.", "—"),
    ],
    "Derivadas": [
        F("Definición de derivada", r"f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}", "h: incremento", "El límite debe existir", "Fundamento de todas las reglas.", "(x²)′=2x"),
        F("Regla de la potencia", r"\frac{d}{dx}x^{n}=nx^{n-1}", "n: constante", "—", "Polinomios y raíces.", "(x⁵)′=5x⁴"),
        F("Regla del producto", r"(uv)'=u'v+uv'", "u, v: funciones", "u, v derivables", "Productos de funciones.", "(x·eˣ)′=eˣ+xeˣ"),
        F("Regla del cociente", r"\left(\frac uv\right)'=\frac{u'v-uv'}{v^{2}}", "u, v: funciones", "v ≠ 0", "Cocientes.", "—"),
        F("Regla de la cadena", r"\frac{d}{dx}f(g(x))=f'(g(x))\,g'(x)", "f externa, g interna", "—", "Funciones compuestas.", "(sen x²)′=2x cos x²"),
        F("Derivadas trigonométricas", r"(\sin x)'=\cos x,\ (\cos x)'=-\sin x,\ (\tan x)'=\sec^{2}x", "x en radianes", "—", "Tabla básica.", "—"),
    ],
    "Integrales": [
        F("Potencia", r"\int x^{n}dx=\frac{x^{n+1}}{n+1}+C", "n ≠ −1", "n ≠ −1", "Polinomios.", "∫x³dx = x⁴/4 + C"),
        F("Logaritmo", r"\int\frac{1}{x}dx=\ln|x|+C", "—", "x ≠ 0", "Integral de 1/x.", "—"),
        F("Sustitución", r"\int f(g(x))g'(x)dx=\int f(u)du,\ u=g(x)", "u: nueva variable", "g derivable", "Integrandos compuestos.", "∫2x cos(x²)dx = sen(x²)+C"),
        F("Integración por partes", r"\int u\,dv=uv-\int v\,du", "u, v: funciones", "u, v derivables; regla LIATE", "Productos de funciones distintas.", "∫x eˣ dx = (x−1)eˣ + C"),
        F("Teorema Fundamental del Cálculo", r"\int_a^b f(x)dx=F(b)-F(a)", "F′=f", "f continua en [a,b]", "Evaluar integrales definidas.", "∫₀¹ x² dx = 1/3"),
        F("Integral impropia", r"\int_a^{\infty}f\,dx=\lim_{t\to\infty}\int_a^t f\,dx", "t: variable auxiliar", "El límite debe ser finito", "Intervalos infinitos o integrandos no acotados.", "∫₁^∞ 1/x² dx = 1"),
    ],
    "Integrales múltiples": [
        F("Teorema de Fubini", r"\iint_R f\,dA=\int_a^b\int_{g_1(x)}^{g_2(x)}f\,dy\,dx", "R: región tipo I", "f continua en R", "Convertir integrales dobles en iteradas.", "—"),
        F("Cambio de variables", r"\iint_R f\,dA=\iint_S f(x(u,v),y(u,v))\left|\frac{\partial(x,y)}{\partial(u,v)}\right|du\,dv", "J: Jacobiano", "Transformación inyectiva", "Regiones con otra simetría.", "—"),
        F("Volumen bajo una superficie", r"V=\iint_R f(x,y)\,dA", "f ≥ 0 en R", "—", "Volúmenes.", "—"),
    ],
    "Coordenadas polares": [
        F("Conversión", r"x=r\cos\theta,\ y=r\sin\theta,\ r^{2}=x^{2}+y^{2},\ \tan\theta=\frac yx", "r ≥ 0, θ ∈ [0,2π)", "—", "Regiones circulares.", "(1,1) → (√2, π/4)"),
        F("Elemento de área", r"dA=r\,dr\,d\theta", "—", "—", "Integrales dobles en polares.", "Área del círculo = ∫₀^{2π}∫₀^R r dr dθ = πR²"),
        F("Área en polares", r"A=\frac12\int_\alpha^\beta r(\theta)^{2}d\theta", "r(θ): curva", "r(θ) ≥ 0", "Áreas de rosas, cardioides.", "—"),
    ],
    "Coordenadas cilíndricas": [
        F("Conversión", r"x=r\cos\theta,\ y=r\sin\theta,\ z=z", "r ≥ 0, θ ∈ [0,2π)", "—", "Cilindros, conos, paraboloides.", "—"),
        F("Elemento de volumen", r"dV=r\,dz\,dr\,d\theta", "—", "—", "Integrales triples con simetría axial.", "V del cilindro = πR²h"),
    ],
    "Coordenadas esféricas": [
        F("Conversión", r"x=\rho\sin\varphi\cos\theta,\ y=\rho\sin\varphi\sin\theta,\ z=\rho\cos\varphi", "ρ ≥ 0, θ ∈ [0,2π), φ ∈ [0,π]", "Convención: φ medido desde +z", "Esferas, conos.", "—"),
        F("Elemento de volumen", r"dV=\rho^{2}\sin\varphi\,d\rho\,d\varphi\,d\theta", "—", "—", "Integrales triples con simetría esférica.", "V esfera = 4πR³/3"),
    ],
    "Cálculo vectorial": [
        F("Gradiente", r"\nabla f=\left(\frac{\partial f}{\partial x},\frac{\partial f}{\partial y},\frac{\partial f}{\partial z}\right)", "f: campo escalar", "f diferenciable", "Dirección de máximo crecimiento.", "∇(x²+y²)=(2x,2y)"),
        F("Divergencia", r"\nabla\cdot\mathbf F=\frac{\partial F_1}{\partial x}+\frac{\partial F_2}{\partial y}+\frac{\partial F_3}{\partial z}", "F: campo vectorial", "F diferenciable", "Fuentes y sumideros.", "—"),
        F("Rotacional", r"\nabla\times\mathbf F=\begin{vmatrix}\mathbf i&\mathbf j&\mathbf k\\\partial_x&\partial_y&\partial_z\\F_1&F_2&F_3\end{vmatrix}", "F: campo vectorial", "F diferenciable", "Circulación local; campos conservativos (∇×F=0).", "—"),
        F("Integral de línea de un campo", r"\int_C\mathbf F\cdot d\mathbf r=\int_a^b\mathbf F(\mathbf r(t))\cdot\mathbf r'(t)\,dt", "r(t): parametrización de C", "C suave a trozos", "Trabajo.", "—"),
        F("Teorema de Green", r"\oint_C P\,dx+Q\,dy=\iint_R\left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right)dA", "C: curva cerrada positiva", "P, Q con derivadas continuas", "Relaciona línea y área.", "—"),
        F("Teorema de la divergencia", r"\iint_S\mathbf F\cdot d\mathbf S=\iiint_E\nabla\cdot\mathbf F\,dV", "S: frontera de E, orientada hacia afuera", "F de clase C¹", "Flujo a través de superficies cerradas.", "—"),
    ],
    "Ecuaciones diferenciales": [
        F("Lineal de primer orden", r"y'+P(x)y=Q(x)\Rightarrow y=e^{-\int P}\left(\int Qe^{\int P}dx+C\right)", "P, Q: funciones de x", "P, Q continuas", "Circuitos RC, mezclas.", "y′+y=x"),
        F("Separable", r"\frac{dy}{dx}=g(x)h(y)\Rightarrow\int\frac{dy}{h(y)}=\int g(x)dx", "g, h: funciones", "h(y) ≠ 0", "Crecimiento exponencial.", "y′=ky → y=Ce^{kx}"),
        F("Lineal homogénea de 2.º orden", r"ay''+by'+cy=0\Rightarrow ar^{2}+br+c=0", "r: raíz característica", "a ≠ 0, coef. constantes", "Vibraciones, circuitos RLC.", "y″−y=0 → y=C₁eˣ+C₂e⁻ˣ"),
    ],
    "Métodos numéricos": [
        F("Newton–Raphson", r"x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}", "x₀: semilla", "f′(xₙ) ≠ 0; buena semilla", "Raíces de ecuaciones.", "√2: f=x²−2"),
        F("Bisección", r"c=\frac{a+b}{2}", "[a,b]: intervalo con f(a)f(b)<0", "f continua", "Método robusto de raíces.", "—"),
        F("Regla del trapecio", r"\int_a^b f\,dx\approx\frac h2\left[f(x_0)+2\sum f(x_i)+f(x_n)\right]", "h=(b−a)/n", "f continua", "Integración numérica.", "—"),
        F("Regla de Simpson", r"\int_a^b f\,dx\approx\frac h3\left[f_0+4f_1+2f_2+\dots+4f_{n-1}+f_n\right]", "n par; h=(b−a)/n", "f de clase C⁴", "Integración numérica precisa.", "—"),
        F("Polinomio de Lagrange", r"P(x)=\sum_{i}y_i\prod_{j\ne i}\frac{x-x_j}{x_i-x_j}", "(xᵢ,yᵢ): datos", "xᵢ distintos", "Interpolación.", "—"),
    ],
}
