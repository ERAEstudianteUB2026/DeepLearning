"""Laboratorio didáctico RNN 2-2-1: forward y BPTT manual."""
import time
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Laboratorio RNN y BPTT", page_icon="🧮", layout="wide")
st.title("Laboratorio interactivo de RNN y BPTT")
st.caption("Red 2–2–1 · tres tiempos · derivadas y actualizaciones calculadas manualmente")

DEFAULTS = {
    "Wx": [[.2,.1],[.1,.2]], "Wh": [[.5,.1],[.1,.5]],
    "Wy": [[.4,.6]], "h0": [0.,0.],
    "X": [[1.,0.],[0.,1.],[1.,1.]], "target": .8, "eta": .05,
}

def fnum(v):
    return f"{float(v):.6f}"

def matrix_text(m):
    return "\\begin{pmatrix}" + "\\\\".join(" & ".join(fnum(x) for x in row) for row in np.atleast_2d(m)) + "\\end{pmatrix}"

def as_matrix(text, shape, label):
    try:
        a=np.array([[float(z.strip()) for z in line.split(',')] for line in text.strip().splitlines()],dtype=float)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f"{label} debe tener dimensiones {shape} y números finitos")
        return a
    except Exception as ex:
        raise ValueError(f"Revisa {label}: escribe una fila por línea y separa los valores por comas. {ex}") from ex

def as_text(a):
    return "\n".join(", ".join(str(x) for x in row) for row in np.atleast_2d(a))

class RNNManual:
    def __init__(self, Wx, Wh, Wy):
        self.Wx=Wx.copy();self.Wh=Wh.copy();self.Wy=Wy.copy()
    def forward(self, X, h0, target):
        hs=[h0.copy()]; aa=[]
        for x in X:
            a=self.Wx@x+self.Wh@hs[-1]
            h=np.tanh(a)
            aa.append(a);hs.append(h)
        yhat=float((self.Wy@hs[-1])[0]); error=yhat-target
        return {"a":aa,"h":hs,"yhat":yhat,"error":error,"loss":.5*error**2}

class BPTTManual:
    @staticmethod
    def backward(net, X, result):
        d=[None]*len(X); gWx=np.zeros_like(net.Wx);gWh=np.zeros_like(net.Wh)
        gWy=result["error"]*result["h"][-1].reshape(1,-1)
        for t in range(len(X)-1,-1,-1):
            upstream=(net.Wy.T[:,0]*result["error"] if t==len(X)-1 else net.Wh.T@d[t+1])
            d[t]=upstream*(1-result["h"][t+1]**2)
            gWx+=np.outer(d[t],X[t]);gWh+=np.outer(d[t],result["h"][t])
        return {"delta":d,"Wx":gWx,"Wh":gWh,"Wy":gWy}
    @staticmethod
    def update(net, grad, eta):
        return RNNManual(net.Wx-eta*grad["Wx"],net.Wh-eta*grad["Wh"],net.Wy-eta*grad["Wy"])

def build_steps(net,X,h0,target,result,grad,eta):
    steps=[]
    def add(section,title,formula,substitution,result_text,explanation):
        for phase,body in [("Fórmula",formula),("Sustitución",substitution),("Resultado",result_text)]:
            steps.append({"section":section,"title":title,"phase":phase,"latex":body,"explanation":explanation})
    for t in range(3):
        x=X[t];hp=result['h'][t];a=result['a'][t];h=result['h'][t+1]
        for i in range(2):
            add('Forward',f'Tiempo {t+1}: a_{{{i+1},{t+1}}}',
                rf'a_{{{i+1},{t+1}}}=\sum_{{j=1}}^2 w^x_{{{i+1}j}}x_{{j,{t+1}}}+\sum_{{j=1}}^2 w^h_{{{i+1}j}}h_{{j,{t}}}',
                rf'a_{{{i+1},{t+1}}}={fnum(net.Wx[i,0])}({fnum(x[0])})+{fnum(net.Wx[i,1])}({fnum(x[1])})+{fnum(net.Wh[i,0])}({fnum(hp[0])})+{fnum(net.Wh[i,1])}({fnum(hp[1])})',
                rf'a_{{{i+1},{t+1}}}={fnum(a[i])}',
                'Suma de la contribución de entrada y la contribución recurrente.')
        for i in range(2):
            add('Forward',f'Tiempo {t+1}: h_{{{i+1},{t+1}}}',
                rf'h_{{{i+1},{t+1}}}=\tanh(a_{{{i+1},{t+1}}})',
                rf'h_{{{i+1},{t+1}}}=\tanh({fnum(a[i])})',
                rf'h_{{{i+1},{t+1}}}={fnum(h[i])}',
                'La activación tanh se aplica por separado a cada neurona oculta.')
    # Vista matricial completa de cada instante, sin sustituir pasos escalares.
    # Se ubican tras los pasos escalares de su instante en una segunda pasada.
    matrix_steps=[]
    for t in range(3):
        x=X[t]; hp=result['h'][t]; a=result['a'][t]; h_t=result['h'][t+1]
        base=rf'\mathbf a_{{{t+1}}}=W_x\mathbf x_{{{t+1}}}+W_h\mathbf h_{{{t}}}'
        numeric=rf'\mathbf a_{{{t+1}}}={matrix_text(net.Wx)}{matrix_text(x.reshape(-1,1))}+{matrix_text(net.Wh)}{matrix_text(hp.reshape(-1,1))}'
        part1=net.Wx@x;part2=net.Wh@hp
        expanded=rf'\mathbf a_{{{t+1}}}={matrix_text(part1.reshape(-1,1))}+{matrix_text(part2.reshape(-1,1))}={matrix_text(a.reshape(-1,1))}'
        for phase,latex in [('Fórmula matricial',base),('Sustitución matricial',numeric),('Producto matricial y resultado',expanded)]:
            matrix_steps.append(dict(section='Forward',title=f'Tiempo {t+1}: preactivación matricial',phase=phase,latex=latex,explanation='Las filas de las matrices producen las preactivaciones de las dos neuronas.'))
        for phase,latex in [('Fórmula matricial',rf'\mathbf h_{{{t+1}}}=\tanh(\mathbf a_{{{t+1}}})'),('Sustitución matricial',rf'\mathbf h_{{{t+1}}}=\tanh\left({matrix_text(a.reshape(-1,1))}\right)'),('Resultado matricial',rf'\mathbf h_{{{t+1}}}={matrix_text(h_t.reshape(-1,1))}')]:
            matrix_steps.append(dict(section='Forward',title=f'Tiempo {t+1}: activación matricial',phase=phase,latex=latex,explanation='La función tanh se aplica componente a componente.'))
    # Conservar el orden: cada bloque matricial sigue al bloque escalar de su instante.
    original_forward=[z for z in steps if z['section']=='Forward']
    reordered=[]
    for t in range(1,4):
        reordered.extend(z for z in original_forward if z['title'].startswith(f'Tiempo {t}:'))
        reordered.extend(z for z in matrix_steps if z['title'].startswith(f'Tiempo {t}:'))
    steps[:]=reordered
    h=result['h'][-1]
    add('Forward','Predicción final',r'\hat y_3=w^y_{11}h_{1,3}+w^y_{12}h_{2,3}',
        rf'\hat y_3={fnum(net.Wy[0,0])}({fnum(h[0])})+{fnum(net.Wy[0,1])}({fnum(h[1])})',
        rf'\hat y_3={fnum(result["yhat"])}','La salida es lineal y se calcula únicamente en el instante final.')
    add('Forward','Predicción matricial',r'\hat y_3=W_y\mathbf h_3',
        rf'\hat y_3={matrix_text(net.Wy)}{matrix_text(h.reshape(-1,1))}',
        rf'\hat y_3={fnum(result["yhat"])}','La capa de salida combina ambos estados ocultos con los pesos W_y.')
    add('Forward','Pérdida',r'\mathcal L=\frac12(\hat y_3-y)^2',
        rf'\mathcal L=\frac12({fnum(result["yhat"])}-{fnum(target)})^2',
        rf'\mathcal L={fnum(result["loss"])}','La pérdida depende solo de la predicción en t=3.')
    add('Backward','Derivada de salida',r'e_3=\frac{\partial\mathcal L}{\partial\hat y_3}=\hat y_3-y',
        rf'e_3={fnum(result["yhat"])}-{fnum(target)}',rf'e_3={fnum(result["error"])}',
        'Es el punto de partida de la propagación hacia atrás.')
    for t in (2,1,0):
        d=grad['delta'][t]; h=result['h'][t+1]
        upstream=net.Wy.T[:,0]*result['error'] if t==2 else net.Wh.T@grad['delta'][t+1]
        for i in range(2):
            if t==2:
                formula=rf'\delta_{{{i+1},3}}=w^y_{{1,{i+1}}}e_3(1-h_{{{i+1},3}}^2)'
                subs=rf'\delta_{{{i+1},3}}={fnum(net.Wy[0,i])}({fnum(result["error"])})\left(1-{fnum(h[i])}^2\right)'
            else:
                formula=rf'\delta_{{{i+1},{t+1}}}=\left(\sum_{{j=1}}^2 w^h_{{j,{i+1}}}\delta_{{j,{t+2}}}\right)(1-h_{{{i+1},{t+1}}}^2)'
                subs=rf'\delta_{{{i+1},{t+1}}}=\left({fnum(net.Wh[0,i])}({fnum(grad["delta"][t+1][0])})+{fnum(net.Wh[1,i])}({fnum(grad["delta"][t+1][1])})\right)(1-{fnum(h[i])}^2)'
            add('Backward',f'Delta en t={t+1}, neurona {i+1}',formula,subs,
                rf'\delta_{{{i+1},{t+1}}}={fnum(d[i])}',
                'Se usa la regla de la cadena y la derivada tanh\u2032(a)=1−tanh²(a).')
    # Complementos matriciales de retropropagación; se conservan todos los deltas escalares.
    delta_matrix_steps=[]
    for t in (2,1,0):
        hvec=result['h'][t+1];dvec=grad['delta'][t]
        derivative=(1-hvec*hvec).reshape(-1,1)
        if t==2:
            formula=r'\boldsymbol\delta_3=(W_y^\top e_3)\odot(\mathbf 1-\mathbf h_3\odot\mathbf h_3)'
            source=net.Wy.T;right=np.array([[result['error']]])
        else:
            formula=rf'\boldsymbol\delta_{{{t+1}}}=(W_h^\top\boldsymbol\delta_{{{t+2}}})\odot(\mathbf 1-\mathbf h_{{{t+1}}}\odot\mathbf h_{{{t+1}}})'
            source=net.Wh.T;right=grad['delta'][t+1].reshape(-1,1)
        upstream=source@right
        numeric=rf'\boldsymbol\delta_{{{t+1}}}=\left({matrix_text(source)}{matrix_text(right)}\right)\odot{matrix_text(derivative)}'
        resolved=rf'\boldsymbol\delta_{{{t+1}}}={matrix_text(upstream)}\odot{matrix_text(derivative)}={matrix_text(dvec.reshape(-1,1))}'
        for phase,latex in [('Fórmula matricial',formula),('Sustitución matricial',numeric),('Producto matricial y resultado',resolved)]:
            delta_matrix_steps.append(dict(section='Backward',title=f'Delta matricial en t={t+1}',phase=phase,latex=latex,explanation='Se aplica la regla de la cadena, la transposición y el producto de Hadamard.'))
    for t in (3,2,1):
        index=max(i for i,z in enumerate(steps) if z['section']=='Backward' and z['title'].startswith(f'Delta en t={t},'))
        inserts=[z for z in delta_matrix_steps if z['title']==f'Delta matricial en t={t}']
        steps[index+1:index+1]=inserts
    for key,formula,explanation in [
        ('Wy',r'\nabla_{W_y}\mathcal L=e_3h_3^\top','La matriz de salida contribuye únicamente en t=3.'),
        ('Wx',r'\nabla_{W_x}\mathcal L=\sum_{t=1}^{3}\delta_tx_t^\top','Los mismos pesos de entrada se utilizan en los tres tiempos.'),
        ('Wh',r'\nabla_{W_h}\mathcal L=\sum_{t=1}^{3}\delta_th_{t-1}^\top','Los pesos recurrentes se comparten y sus contribuciones se suman.')]:
        if key!='Wy':
            for t in range(3):
                left=X[t] if key=='Wx' else result['h'][t]
                contribution=np.outer(grad['delta'][t],left)
                add('Backward',f'Contribución a ∇{key} en t={t+1}',
                    rf'G_{{{key},{t+1}}}=\delta_{{{t+1}}}'+(rf'x_{{{t+1}}}^\top' if key=='Wx' else rf'h_{{{t}}}^\top'),
                    rf'G_{{{key},{t+1}}}={matrix_text(grad["delta"][t].reshape(-1,1))}{matrix_text(left.reshape(1,-1))}',
                    rf'G_{{{key},{t+1}}}={matrix_text(contribution)}',explanation)
        add('Backward',f'Gradiente total de {key}',formula,
            rf'\nabla_{{{key}}}\mathcal L={matrix_text(grad[key])}',
            rf'\nabla_{{{key}}}\mathcal L={matrix_text(grad[key])}',explanation)
    for key in ('Wx','Wh','Wy'):
        old=getattr(net,key);new=old-eta*grad[key]
        add('Actualización',f'Actualizar {key}',
            rf'{key}^{{(k+1)}}={key}^{{(k)}}-\eta\nabla_{{{key}}}\mathcal L',
            rf'{key}^{{(k+1)}}={matrix_text(old)}-{fnum(eta)}{matrix_text(grad[key])}',
            rf'{key}^{{(k+1)}}={matrix_text(new)}',
            'La actualización usa descenso del gradiente, sin diferenciación automática.')
    return steps

# Capa visual adicional: conserva íntegramente los pasos matemáticos originales.
import re
import html

def _visual_context(step):
    section=step['section']; title=step['title']
    m=re.search(r'(?:Tiempo |t=)([123])',title)
    t=int(m.group(1)) if m else (3 if section=='Forward' else 1)
    neuron=None
    match=re.search(r'a_\{([12]),|h_\{([12]),|neurona ([12])',title)
    if match:
        neuron=int(next(v for v in match.groups() if v is not None))
    return section,t,neuron

def _network_svg(step):
    """Arquitectura completa y gradientes: conexiones reales entre estados ocultos."""
    section, t, neuron = _visual_context(step)
    backward = section == 'Backward'
    updating = section == 'Actualización'
    title = step['title']
    teal, orange, gray = '#087f78', '#c66c14', '#93a4b5'
    active = orange if backward else teal
    ink = '#263647'
    pieces = [
        '<svg viewBox="0 0 1600 345" xmlns="http://www.w3.org/2000/svg" '
        'role="img" aria-label="RNN 2-2-1 desplegada: entradas, estados ocultos, '
        'conexiones recurrentes, salida y pérdida" style="width:100%;height:auto">',
        '<defs>',
        '<marker id="bptt_fwd" markerWidth="7" markerHeight="7" refX="6" refY="3" '
        'orient="auto"><path d="M0,0 L6,3 L0,6" fill="#087f78"/></marker>',
        '<marker id="bptt_back" markerWidth="7" markerHeight="7" refX="6" refY="3" '
        'orient="auto"><path d="M0,0 L6,3 L0,6" fill="#c66c14"/></marker>',
        '</defs>'
    ]
    centers = [270, 630, 990]
    ys = [207, 264]
    # Marco de cada instante. Los nodos x están arriba y los nodos h abajo,
    # para que las conexiones recurrentes no terminen visualmente en las entradas.
    for k, cx in enumerate(centers, start=1):
        selected = (k == t) or updating
        border = active if selected else gray
        pieces.append(f'<rect x="{cx-122}" y="29" width="244" height="278" rx="15" '
                      f'fill="{active}" fill-opacity="{.065 if selected else .014}" '
                      f'stroke="{border}" stroke-width="{3 if selected else 1.4}"/>')
        pieces.append(f'<text x="{cx}" y="56" text-anchor="middle" fill="{ink}" '
                      f'font-size="18" font-weight="700">Instante t = {k}</text>')
        pieces.append(f'<text x="{cx-79}" y="80" text-anchor="middle" '
                      f'fill="{ink}" font-size="13">Entrada x{k}</text>')
        pieces.append(f'<text x="{cx+80}" y="80" text-anchor="middle" '
                      f'fill="{ink}" font-size="13">Estado h{k}</text>')
        # Conexiones Wx: las dos entradas alcanzan ambas neuronas ocultas.
        for i, yi in enumerate((117, 160), start=1):
            for j, yj in enumerate(ys, start=1):
                highlight = selected and (neuron is None or neuron == j) and not backward
                color = teal if highlight else gray
                pieces.append(f'<line x1="{cx-61}" y1="{yi}" x2="{cx+55}" y2="{yj}" '
                              f'stroke="{color}" stroke-width="{2.8 if highlight else 1.2}" '
                              f'stroke-opacity="{1 if highlight else .45}"/>')
        for j, y in enumerate((117, 160), start=1):
            pieces.append(f'<circle cx="{cx-79}" cy="{y}" r="19" fill="#e2e8f0" '
                          f'stroke="{gray}" stroke-width="1.5"/>')
            pieces.append(f'<text x="{cx-79}" y="{y+5}" text-anchor="middle" '
                          f'fill="{ink}" font-size="15">x{j}</text>')
        for j, y in enumerate(ys, start=1):
            hi = selected and (neuron is None or neuron == j)
            col = active if hi else gray
            pieces.append(f'<circle cx="{cx+80}" cy="{y}" r="25" '
                          f'fill="{col}" fill-opacity="{.23 if hi else .10}" '
                          f'stroke="{col}" stroke-width="{3.4 if hi else 1.4}"/>')
            pieces.append(f'<text x="{cx+80}" y="{y+5}" text-anchor="middle" '
                          f'fill="{ink}" font-size="17" font-weight="600">h{j}</text>')
        pieces.append(f'<text x="{cx}" y="293" text-anchor="middle" '
                      f'fill="{ink}" font-size="13">Wₓ · tanh</text>')
    # h0 es un estado inicial, no una entrada x1.
    pieces.append(f'<text x="51" y="166" text-anchor="middle" fill="{ink}" '
                  f'font-size="16" font-weight="700">Estado h₀</text>')
    for j, y in enumerate(ys, start=1):
        pieces.append(f'<circle cx="51" cy="{y}" r="22" fill="#e2e8f0" '
                      f'stroke="{gray}" stroke-width="1.7"/>')
        pieces.append(f'<text x="51" y="{y+5}" text-anchor="middle" '
                      f'fill="{ink}" font-size="15">h₀,{j}</text>')
        color = teal if section == 'Forward' and t == 1 else gray
        for dest in ys:
            pieces.append(f'<line x1="73" y1="{y}" x2="{centers[0]+54}" y2="{dest}" '
                          f'stroke="{color}" stroke-opacity="{.85 if color==teal else .27}" '
                          f'stroke-width="{2 if color==teal else 1.1}"/>')
    pieces.append(f'<text x="112" y="187" fill="{ink}" font-size="13">Wₕ</text>')
    # Cuatro conexiones Wh entre h(t-1) y h(t), nunca entre nodos de entrada.
    for k in (1, 2):
        src = centers[k-1] + 80
        dst = centers[k] + 80
        selected_edge = (backward and t == k) or (not backward and t == k+1)
        if updating:
            selected_edge = False
        for i, y1 in enumerate(ys, start=1):
            for j, y2 in enumerate(ys, start=1):
                highlight = selected_edge and (neuron is None or (i if backward else j) == neuron)
                color = active if highlight else gray
                x1, yy1, x2, yy2 = (dst-26,y2,src+26,y1) if backward else (src+26,y1,dst-26,y2)
                pieces.append(f'<line x1="{x1}" y1="{yy1}" x2="{x2}" y2="{yy2}" '
                              f'stroke="{color}" stroke-opacity="{.9 if highlight else .35}" '
                              f'stroke-width="{3.1 if highlight else 1.3}" '
                              + (f'marker-end="url(#{"bptt_back" if backward else "bptt_fwd"})"' if highlight else '') + '/>')
        pieces.append(f'<text x="{(src+dst)/2}" y="187" text-anchor="middle" '
                      f'fill="{active if selected_edge else ink}" font-size="16" font-weight="600">'
                      f'{"Wₕᵀ · gradiente" if backward else "Wₕ"}</text>')
    # Salida lineal y pérdida. En backward las flechas invierten su dirección.
    is_output = ('Predicción' in title or 'Derivada de salida' in title or
                 (backward and t == 3))
    is_loss = ('Pérdida' in title or 'Derivada de salida' in title)
    out_color = orange if backward else teal
    for y in ys:
        if backward:
            x1,y1,x2,y2 = 1220,235,1097,y
        else:
            x1,y1,x2,y2 = 1097,y,1220,235
        pieces.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                      f'stroke="{out_color if is_output else gray}" '
                      f'stroke-width="{3.4 if is_output else 1.5}" '
                      f'stroke-opacity="{1 if is_output else .45}" '
                      + (f'marker-end="url(#{"bptt_back" if backward else "bptt_fwd"})"' if is_output else '') + '/>')
    pieces.append(f'<text x="1160" y="188" text-anchor="middle" fill="{ink}" '
                  f'font-size="17">{"Wᵧᵀ" if backward else "Wᵧ"}</text>')
    pieces.append(f'<circle cx="1247" cy="235" r="29" fill="{out_color}" '
                  f'fill-opacity="{.24 if is_output else .08}" '
                  f'stroke="{out_color if is_output else gray}" '
                  f'stroke-width="{3 if is_output else 1.5}"/>')
    pieces.append(f'<text x="1247" y="241" text-anchor="middle" '
                  f'fill="{ink}" font-size="19">ŷ₃</text>')
    loss_color = orange if backward else teal
    if backward:
        x1,x2=1385,1278
    else:
        x1,x2=1276,1385
    pieces.append(f'<line x1="{x1}" y1="235" x2="{x2}" y2="235" '
                  f'stroke="{loss_color if is_loss else gray}" '
                  f'stroke-width="{3 if is_loss else 1.4}" '
                  + (f'marker-end="url(#{"bptt_back" if backward else "bptt_fwd"})"' if is_loss else '') + '/>')
    pieces.append(f'<rect x="1388" y="202" width="174" height="66" rx="13" '
                  f'fill="{loss_color}" fill-opacity="{.18 if is_loss else .06}" '
                  f'stroke="{loss_color if is_loss else gray}" '
                  f'stroke-width="{3 if is_loss else 1.5}"/>')
    pieces.append(f'<text x="1475" y="242" text-anchor="middle" '
                  f'fill="{ink}" font-size="18" font-weight="600">Pérdida L</text>')
    caption = ('Backward: derivadas desde L hasta h₃, h₂ y h₁ (Wₕᵀ)' if backward
               else 'Actualización de los pesos compartidos' if updating
               else 'Forward: estados h₀ → h₁ → h₂ → h₃ → ŷ₃ → L')
    pieces.append(f'<text x="800" y="332" text-anchor="middle" fill="{ink}" '
                  f'font-size="17" font-weight="600">{caption}</text>')
    pieces.append('</svg>')
    return ''.join(pieces)

def show_visual_step(step, result):
    section,t,neuron=_visual_context(step)
    st.markdown('#### Red desplegada: ¿dónde ocurre esta operación?')
    st.markdown(_network_svg(step),unsafe_allow_html=True)
    if section=='Forward':
        if 'Predicción' in step['title']:
            st.info('La salida combina los dos estados ocultos del último instante; solo en t=3 se calcula la predicción.')
        elif 'Pérdida' in step['title']:
            st.info('La pérdida compara la predicción con el objetivo. Este error inicia el backward.')
        else:
            st.info(f'En t={t}, las entradas xₜ y el estado hₜ₋₁ alimentan las dos neuronas ocultas. Se destaca la neurona del cálculo actual.')
    elif section=='Backward':
        st.info(f'El error retrocede por las conexiones recurrentes. En t={t} se aplica la regla de la cadena y los pesos compartidos reciben contribuciones de distintos instantes.')
    else:
        st.info('Los gradientes acumulados permiten corregir cada matriz de pesos mediante descenso del gradiente.')
    if section=='Forward' and step['title'].startswith('Tiempo') and ': h_' in step['title']:
        with st.expander('Ver la función tanh y el punto de activación',expanded=False):
            z=np.linspace(-3,3,121)
            st.line_chart(pd.DataFrame({'tanh(z)':np.tanh(z)},index=z),x_label='Preactivación z',y_label='tanh(z)')
            st.caption('La función tanh transforma las preactivaciones al intervalo (-1,1).')

with st.sidebar:
    st.header('Datos del ejercicio')
    st.caption('Una fila por línea; valores separados por comas.')
    wx_text=st.text_area('Wx (2×2)',as_text(DEFAULTS['Wx']),height=75)
    wh_text=st.text_area('Wh (2×2)',as_text(DEFAULTS['Wh']),height=75)
    wy_text=st.text_area('Wy (1×2)',as_text(DEFAULTS['Wy']),height=55)
    h0_text=st.text_input('h0 (2 valores)', '0, 0')
    x_text=st.text_area('X: x1, x2, x3 (3×2)',as_text(DEFAULTS['X']),height=95)
    target=st.number_input('Objetivo y',value=.8,format='%.4f')
    eta=st.number_input('Tasa de aprendizaje η',min_value=.000001,value=.05,format='%.5f')
    if st.button('Cargar / reiniciar el ejercicio',type='primary',use_container_width=True):
        try:
            wx=as_matrix(wx_text,(2,2),'Wx');wh=as_matrix(wh_text,(2,2),'Wh')
            wy=as_matrix(wy_text,(1,2),'Wy');h0=as_matrix(h0_text,(1,2),'h0')[0]
            X=as_matrix(x_text,(3,2),'X')
            st.session_state.net=RNNManual(wx,wh,wy)
            st.session_state.X=X;st.session_state.h0=h0
            st.session_state.target=float(target);st.session_state.eta=float(eta)
            st.session_state.step=0;st.session_state.epoch=1;st.session_state.history=[]
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

if 'net' not in st.session_state:
    st.session_state.net=RNNManual(np.array(DEFAULTS['Wx']),np.array(DEFAULTS['Wh']),np.array(DEFAULTS['Wy']))
    st.session_state.X=np.array(DEFAULTS['X']);st.session_state.h0=np.array(DEFAULTS['h0'])
    st.session_state.target=.8;st.session_state.eta=.05
    st.session_state.step=0;st.session_state.epoch=1;st.session_state.history=[]

net=st.session_state.net;X=st.session_state.X;h0=st.session_state.h0
target=st.session_state.target;eta=st.session_state.eta
result=net.forward(X,h0,target);grad=BPTTManual.backward(net,X,result)
steps=build_steps(net,X,h0,target,result,grad,eta)

with st.expander('📐 Matrices de pesos actuales (se actualizan con cada época)',expanded=True):
    mcols=st.columns(3)
    for col,name in zip(mcols,('Wx','Wh','Wy')):
        with col:
            st.markdown(f'**{name} — {getattr(net,name).shape[0]} × {getattr(net,name).shape[1]}**')
            st.latex(rf'{name}={matrix_text(getattr(net,name))}')
manual, training=st.tabs(['Cálculo paso a paso','Entrenamiento visual'])
with manual:
    st.subheader(f'Época {st.session_state.epoch} · cálculo manual')
    st.caption('Cada clic revela una fórmula, después su sustitución y finalmente su resultado.')
    c1,c2,c3=st.columns([1,1,2])
    with c1:
        if st.button('← Anterior',disabled=st.session_state.step==0,use_container_width=True):
            st.session_state.step-=1;st.rerun()
    with c2:
        if st.button('Siguiente →',disabled=st.session_state.step==len(steps)-1,use_container_width=True):
            st.session_state.step+=1;st.rerun()
    with c3:
        st.progress((st.session_state.step+1)/len(steps),text=f'Paso {st.session_state.step+1} de {len(steps)}')
    st.markdown('**Ir directamente a:**')
    targets=[('Inicio forward',lambda z:z['section']=='Forward'),
             ('t = 1',lambda z:z['section']=='Forward' and z['title'].startswith('Tiempo 1:')),
             ('t = 2',lambda z:z['section']=='Forward' and z['title'].startswith('Tiempo 2:')),
             ('t = 3',lambda z:z['section']=='Forward' and z['title'].startswith('Tiempo 3:')),
             ('Salida final',lambda z:z['section']=='Forward' and z['title']=='Predicción final'),
             ('Inicio backward',lambda z:z['section']=='Backward')]
    navcols=st.columns(6)
    for (label,predicate),column in zip(targets,navcols):
        index=next((i for i,z in enumerate(steps) if predicate(z)),None)
        with column:
            if st.button(label,key='jump_'+label,use_container_width=True,disabled=index is None):
                st.session_state.step=index
                st.rerun()
    current=steps[st.session_state.step]
    show_visual_step(current,result)
    st.markdown(f'<div style="font-size:2rem;font-weight:800;color:{"#c17b25" if current["section"]=="Backward" else "#159c94" if current["section"]=="Forward" else "#5865c8"};padding:10px 0">{html.escape(current["section"])} · {html.escape(current["title"])}</div>',unsafe_allow_html=True)
    st.markdown(f'**{current["phase"]}**')
    st.latex(current['latex'])
    st.write(current['explanation'])
    with st.expander('Historial de pasos ya revelados',expanded=True):
        for i,s in enumerate(steps[:st.session_state.step+1]):
            st.markdown(f'**{i+1}. {s["section"]} — {s["title"]} ({s["phase"]})**')
            st.latex(s['latex'])
    if st.session_state.step==len(steps)-1:
        if st.button('Aplicar actualización y comenzar siguiente época',type='primary'):
            st.session_state.history.append({'Época':st.session_state.epoch,'Predicción':result['yhat'],'Pérdida':result['loss']})
            st.session_state.net=BPTTManual.update(net,grad,eta)
            st.session_state.epoch+=1;st.session_state.step=0;st.rerun()

with training:
    st.subheader('Entrenamiento visual, época por época')
    st.write('El entrenamiento utiliza el mismo forward y el mismo BPTT manual del modo didáctico.')
    n=st.number_input('Número de épocas a ejecutar',min_value=1,max_value=1000,value=50,step=1)
    delay=st.slider('Pausa entre épocas (segundos)',min_value=0.0,max_value=1.0,value=.08,step=.02)
    if st.button('▶ Entrenar y visualizar',type='primary'):
        panel=st.empty();chart=st.empty();weights=st.empty();progress=st.progress(0)
        for k in range(int(n)):
            net=st.session_state.net
            out=net.forward(X,h0,target);g=BPTTManual.backward(net,X,out)
            st.session_state.history.append({'Época':st.session_state.epoch,'Predicción':out['yhat'],'Pérdida':out['loss']})
            st.session_state.net=BPTTManual.update(net,g,eta)
            st.session_state.epoch+=1
            panel.metric('Época completada',st.session_state.epoch-1,delta=f'Pérdida {out["loss"]:.8f}',delta_color='off')
            chart.line_chart(pd.DataFrame(st.session_state.history).set_index('Época')[['Pérdida']])
            new=st.session_state.net
            with weights.container():
                st.markdown('**Backward: gradientes calculados en esta época**')
                st.dataframe(pd.DataFrame(g['Wx']),hide_index=True)
                st.markdown('**Actualización de pesos: antes → después**')
                for name in ('Wx','Wh','Wy'):
                    st.markdown(f'**{name}**')
                    st.latex(rf'{name}^{{nuevo}}={matrix_text(getattr(net,name))}-{fnum(eta)}{matrix_text(g[name])}={matrix_text(getattr(new,name))}')
            progress.progress((k+1)/int(n))
            time.sleep(delay)
        st.session_state.step=0
        st.success('Entrenamiento completado. Puedes volver al modo paso a paso con los pesos actuales.')
    if st.session_state.history:
        st.markdown('**Historial de entrenamiento**')
        st.dataframe(pd.DataFrame(st.session_state.history),hide_index=True,use_container_width=True)
    st.caption('Ejemplo con una única secuencia: el objetivo es ilustrar BPTT, no evaluar generalización.')
