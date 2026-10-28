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
    """RNN 2–2–1 con resaltado EXCLUSIVO de la operación matemática actual."""
    section, t, neuron = _visual_context(step)
    title = step['title']
    backward = section == 'Backward'
    teal, orange, gray, ink = '#087f78', '#c66c14', '#9aaabc', '#263647'
    active = orange if backward else teal
    is_delta = backward and title.startswith('Delta ')
    is_output_derivative = backward and title == 'Derivada de salida'
    is_gwx = backward and ('Contribución a ∇Wx' in title or 'Gradiente total de Wx' in title)
    is_gwh = backward and ('Contribución a ∇Wh' in title or 'Gradiente total de Wh' in title)
    is_gwy = backward and title == 'Gradiente total de Wy'
    is_update = section == 'Actualización'
    forward_hidden = section == 'Forward' and title.startswith('Tiempo ')
    forward_output = section == 'Forward' and 'Predicción' in title
    forward_loss = section == 'Forward' and title == 'Pérdida'
    # No iluminar la salida al calcular delta_2 o delta_1.
    output_edge = forward_output or (is_delta and t == 3) or is_gwy
    loss_edge = forward_loss or is_output_derivative
    # Cada etapa ilumina solo su tramo temporal correspondiente.
    recurrent_edge = ((forward_hidden and t > 1) or
                      (is_delta and t in (1, 2)) or is_gwh)
    input_edge = forward_hidden or is_gwx
    selected_times = ({t} if forward_hidden or is_delta else
                      {t} if ('Contribución a' in title) else set())
    def line(x1,y1,x2,y2,on=False,arrow=False):
        color=active if on else gray
        marker = ' marker-end="url(#back_arrow)"' if arrow and backward else (' marker-end="url(#fwd_arrow)"' if arrow else '')
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{color}" stroke-width="{3.0 if on else 1.25}" '
                f'stroke-opacity="{1 if on else .42}"{marker}/>')
    parts=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1640 345" '
           'role="img" aria-label="RNN desplegada con operación actual resaltada" style="width:100%;height:auto">',
           '<defs><marker id="fwd_arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#087f78"/></marker>'
           '<marker id="back_arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#c66c14"/></marker></defs>']
    centers=[275,650,1025]; hidden_ys=[206,265]; input_ys=[117,160]
    # Conexiones se dibujan ANTES de los nodos para no tapar las etiquetas.
    for k,cx in enumerate(centers,1):
        on=k in selected_times
        parts.append(f'<rect x="{cx-120}" y="25" width="240" height="282" rx="14" fill="{active}" fill-opacity="{.065 if on else .012}" stroke="{active if on else gray}" stroke-width="{2.7 if on else 1.3}"/>')
        parts.append(f'<text x="{cx}" y="52" text-anchor="middle" font-size="18" font-weight="700" fill="{ink}">Instante t = {k}</text>')
        parts.append(f'<text x="{cx-73}" y="78" text-anchor="middle" font-size="13" fill="{ink}">Entrada x{k}</text>')
        parts.append(f'<text x="{cx+76}" y="78" text-anchor="middle" font-size="13" fill="{ink}">Estado h{k}</text>')
        for i,yi in enumerate(input_ys,1):
            for j,yj in enumerate(hidden_ys,1):
                high=input_edge and (k==t or ('Gradiente total de Wx' in title)) and (neuron is None or neuron==j)
                # En ∇Wx, la conexión de entrada es una contribución al gradiente, no una señal hacia adelante.
                parts.append(line(cx-53,yi,cx+50,yj,high))
        for i,yi in enumerate(input_ys,1):
            parts.append(f'<circle cx="{cx-73}" cy="{yi}" r="18" fill="#e2e8f0" stroke="{gray}" stroke-width="1.4"/>')
            parts.append(f'<text x="{cx-73}" y="{yi+5}" text-anchor="middle" font-size="15" fill="{ink}">x{i}</text>')
        for j,yj in enumerate(hidden_ys,1):
            high=(k in selected_times) and (neuron is None or neuron==j)
            col=active if high else gray
            parts.append(f'<circle cx="{cx+76}" cy="{yj}" r="24" fill="{col}" fill-opacity="{.22 if high else .11}" stroke="{col}" stroke-width="{3 if high else 1.4}"/>')
            parts.append(f'<text x="{cx+76}" y="{yj+5}" text-anchor="middle" font-size="17" fill="{ink}">h{j}</text>')
        parts.append(f'<text x="{cx}" y="294" text-anchor="middle" font-size="13" fill="{ink}">Wₓ · tanh</text>')
    parts.append(f'<text x="56" y="160" text-anchor="middle" font-size="16" font-weight="700" fill="{ink}">Estado h₀</text>')
    for i,y in enumerate(hidden_ys,1):
        parts.append(f'<circle cx="56" cy="{y}" r="21" fill="#e2e8f0" stroke="{gray}" stroke-width="1.5"/>')
        parts.append(f'<text x="56" y="{y+5}" text-anchor="middle" font-size="14" fill="{ink}">h₀,{i}</text>')
        for j,yd in enumerate(hidden_ys,1):
            on=forward_hidden and t==1 and (neuron is None or neuron==j)
            parts.append(line(78,y,centers[0]+50,yd,on,on))
    parts.append(f'<text x="120" y="184" font-size="14" fill="{ink}">Wₕ</text>')
    # W_h: cuatro aristas entre estados ocultos; en BPTT se invierte su sentido.
    for k in (1,2):
        left=centers[k-1]+76; right=centers[k]+76
        for i,yi in enumerate(hidden_ys,1):
            for j,yj in enumerate(hidden_ys,1):
                on=recurrent_edge and ((is_gwh and ('Gradiente total' in title or t==k+1)) or
                                      (not is_gwh and ((backward and t==k) or (not backward and t==k+1))))
                on=on and (neuron is None or (i if backward and is_delta else j)==neuron)
                if backward and is_delta:
                    parts.append(line(right-24,yj,left+24,yi,on,on))
                else:
                    parts.append(line(left+24,yi,right-24,yj,on,on))
        parts.append(f'<text x="{(left+right)/2}" y="182" text-anchor="middle" font-size="15" font-weight="600" fill="{active if recurrent_edge and (t==k if backward and is_delta else t==k+1 or is_gwh) else ink}">{"Wₕᵀ · gradiente" if backward and is_delta else "Wₕ"}</text>')
    # Salida: para delta_3 las flechas van de y_hat_3 hacia h_3, NO desde L.
    last=centers[-1]+76
    for y in hidden_ys:
        if backward and (is_delta or is_gwy):
            parts.append(line(1254,235,last+25,y,output_edge,output_edge))
        else:
            parts.append(line(last+25,y,1254,235,output_edge,output_edge))
    parts.append(f'<text x="1190" y="181" text-anchor="middle" font-size="17" fill="{ink}">{"Wᵧᵀ" if backward and is_delta else "Wᵧ"}</text>')
    parts.append(f'<circle cx="1282" cy="235" r="28" fill="{active if output_edge or loss_edge else gray}" fill-opacity=".18" stroke="{active if output_edge or loss_edge else gray}" stroke-width="{3 if output_edge or loss_edge else 1.4}"/>')
    parts.append(f'<text x="1282" y="241" text-anchor="middle" font-size="19" fill="{ink}">ŷ₃</text>')
    if is_output_derivative:
        parts.append(line(1415,235,1312,235,True,True))
    else:
        parts.append(line(1312,235,1415,235,loss_edge,loss_edge))
    parts.append(f'<rect x="1418" y="202" width="190" height="66" rx="13" fill="{active if loss_edge else gray}" fill-opacity=".12" stroke="{active if loss_edge else gray}" stroke-width="{3 if loss_edge else 1.4}"/>')
    parts.append(f'<text x="1513" y="242" text-anchor="middle" font-size="18" font-weight="600" fill="{ink}">Pérdida L</text>')
    if is_delta:
        description = ('Backward: Wᵧᵀ lleva el gradiente desde ŷ₃ hasta h₃' if t==3 else
                       f'Backward: Wₕᵀ lleva el gradiente de t={t+1} a t={t}')
    elif is_output_derivative:
        description='Backward: derivada de L respecto a la predicción ŷ₃'
    elif is_gwx or is_gwh or is_gwy:
        description='Gradiente de pesos: contribuciones por instante y pesos compartidos'
    elif is_update:
        description='Actualización de los pesos con los gradientes acumulados'
    else:
        description='Forward: h₀ → h₁ → h₂ → h₃ → ŷ₃ → L'
    parts.append(f'<text x="820" y="334" text-anchor="middle" font-size="17" font-weight="600" fill="{ink}">{description}</text>')
    parts.append('</svg>')
    return ''.join(parts)

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
        if step['title'] == 'Derivada de salida':
            st.info('Operación actual: se calcula ∂L/∂ŷ₃. Solo se resalta el tramo de la pérdida a la salida.')
        elif step['title'].startswith('Delta '):
            if t == 3:
                st.info('Operación actual: δ₃. Se aplica Wᵧᵀ al error de salida y después la derivada de tanh en h₃. Aún no se retropropaga hacia t=2.')
            else:
                st.info(f'Operación actual: δ{t}. El gradiente llega desde el instante t={t+1} mediante Wₕᵀ y se combina con la derivada local de tanh.')
        elif 'Contribución' in step['title'] or 'Gradiente total' in step['title']:
            st.info('Ahora se calculan gradientes respecto a los pesos compartidos. Se resaltan las conexiones pertinentes; no es un nuevo recorrido temporal de δ.')
        else:
            st.info('Se conserva el procedimiento matemático original y se destaca la operación activa.')
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
