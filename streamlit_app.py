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
    """Ruta matricial completa; cada operación muestra fórmula, sustitución y resultado."""
    steps=[]
    def add(section,title,formula,substitution,answer,explanation,code=None):
        for phase,latex in [('Fórmula matricial',formula),('Sustitución matricial',substitution),('Resultado matricial',answer)]:
            steps.append(dict(section=section,title=title,phase=phase,latex=latex,
                              explanation=explanation,code=code))
    for t in range(3):
        k=t+1; x=X[t]; hp=result['h'][t]; a=result['a'][t]; h=result['h'][t+1]
        ax=net.Wx@x; ah=net.Wh@hp
        add('Forward',f'Tiempo {k}: preactivación matricial',
            rf'\mathbf a_{{{k}}}=W_x\mathbf x_{{{k}}}+W_h\mathbf h_{{{t}}}',
            rf'\mathbf a_{{{k}}}={matrix_text(net.Wx)}{matrix_text(x.reshape(-1,1))}+{matrix_text(net.Wh)}{matrix_text(hp.reshape(-1,1))}',
            rf'\mathbf a_{{{k}}}={matrix_text(ax.reshape(-1,1))}+{matrix_text(ah.reshape(-1,1))}={matrix_text(a.reshape(-1,1))}',
            'Se calculan ambas neuronas mediante productos matriciales.')
        add('Forward',f'Tiempo {k}: activación matricial',
            rf'\mathbf h_{{{k}}}=\tanh(\mathbf a_{{{k}}})',
            rf'\mathbf h_{{{k}}}=\tanh\left({matrix_text(a.reshape(-1,1))}\right)',
            rf'\mathbf h_{{{k}}}={matrix_text(h.reshape(-1,1))}',
            'La función tanh actúa componente a componente.')
    h3=result['h'][3]; e=result['error']
    add('Forward','Predicción matricial',r'\hat y_3=W_y\mathbf h_3',
        rf'\hat y_3={matrix_text(net.Wy)}{matrix_text(h3.reshape(-1,1))}',
        rf'\hat y_3={fnum(result["yhat"])}','La salida es lineal.')
    add('Forward','Pérdida',r'\mathcal L=\frac12(\hat y_3-y)^2',
        rf'\mathcal L=\frac12({fnum(result["yhat"])}-{fnum(target)})^2',
        rf'\mathcal L={fnum(result["loss"])}','Solo hay pérdida en el instante final.')
    add('Backward','B1 · Error de salida',r'\frac{\partial\mathcal L}{\partial\hat y_3}=\hat y_3-y',
        rf'\frac{{\partial\mathcal L}}{{\partial\hat y_3}}={fnum(result["yhat"])}-{fnum(target)}',
        rf'\frac{{\partial\mathcal L}}{{\partial\hat y_3}}={fnum(e)}',
        'Se deriva la pérdida cuadrática respecto de la predicción.','B1')
    gh3=net.Wy.T[:,0]*e
    add('Backward','B2 · Gradiente del estado h3',r'\nabla_{h_3}\mathcal L=W_y^T\frac{\partial\mathcal L}{\partial\hat y_3}',
        rf'\nabla_{{h_3}}\mathcal L={matrix_text(net.Wy.T)}({fnum(e)})',
        rf'\nabla_{{h_3}}\mathcal L={matrix_text(gh3.reshape(-1,1))}',
        'La transpuesta de los pesos de salida lleva el gradiente a h₃.','B2')
    def add_delta(k,upstream,code):
        h=result['h'][k]; D=np.diag(1-h**2); delta=grad['delta'][k-1]
        upstream_name=r'\nabla_{h_3}\mathcal L' if k==3 else rf'W_h^T\delta_{{{k+1}}}'
        add('Backward',f'{code} · Delta matricial en t={k}',
            rf'\delta_{{{k}}}=D_{{{k}}}{upstream_name},\qquad D_{{{k}}}=\operatorname{{diag}}\left(1-h_{{{k},1}}^2,1-h_{{{k},2}}^2\right)',
            rf'\delta_{{{k}}}={matrix_text(D)}{matrix_text(upstream.reshape(-1,1))}',
            rf'\delta_{{{k}}}={matrix_text(delta.reshape(-1,1))}',
            'Dₜ es el jacobiano de tanh evaluado en aₜ. Se multiplica por el gradiente que llega desde la derecha.',code)
    add_delta(3,gh3,'B3')
    gwy=e*h3.reshape(1,-1)
    add('Backward','B4 · Gradiente de Wy',r'\nabla_{W_y}\mathcal L=\frac{\partial\mathcal L}{\partial\hat y_3}h_3^T',
        rf'\nabla_{{W_y}}\mathcal L=({fnum(e)}){matrix_text(h3.reshape(1,-1))}',
        rf'\nabla_{{W_y}}\mathcal L={matrix_text(gwy)}',
        'Un producto exterior determina los gradientes de los dos pesos de salida.','B4')
    for k,code_h,code_delta in [(2,'B5','B6'),(1,'B7','B8')]:
        upstream=net.Wh.T@grad['delta'][k]
        add('Backward',f'{code_h} · Gradiente del estado h{k}',
            rf'\nabla_{{h_{{{k}}}}}\mathcal L=W_h^T\delta_{{{k+1}}}',
            rf'\nabla_{{h_{{{k}}}}}\mathcal L={matrix_text(net.Wh.T)}{matrix_text(grad["delta"][k].reshape(-1,1))}',
            rf'\nabla_{{h_{{{k}}}}}\mathcal L={matrix_text(upstream.reshape(-1,1))}',
            'El gradiente retrocede por los pesos recurrentes transpuestos.',code_h)
        add_delta(k,upstream,code_delta)
    for key,code,left_name in [('Wh','B9','h'),('Wx','B10','x')]:
        pieces=[]
        for t in range(3):
            left=result['h'][t] if key=='Wh' else X[t]
            piece=np.outer(grad['delta'][t],left);pieces.append(piece)
            add('Backward',f'{code} · Contribución en t={t+1}',
                rf'G_{{{left_name}}}^{{({t+1})}}=\delta_{{{t+1}}}{left_name}_{{{t if key=="Wh" else t+1}}}^T',
                rf'G_{{{left_name}}}^{{({t+1})}}={matrix_text(grad["delta"][t].reshape(-1,1))}{matrix_text(left.reshape(1,-1))}',
                rf'G_{{{left_name}}}^{{({t+1})}}={matrix_text(piece)}',
                'Cada aparición de los pesos compartidos aporta una matriz 2×2.',code)
        add('Backward',f'{code} · Gradiente total de {key}',
            rf'\nabla_{{W_{left_name}}}\mathcal L=G_{{{left_name}}}^{{(1)}}+G_{{{left_name}}}^{{(2)}}+G_{{{left_name}}}^{{(3)}}',
            rf'\nabla_{{W_{left_name}}}\mathcal L={matrix_text(pieces[0])}+{matrix_text(pieces[1])}+{matrix_text(pieces[2])}',
            rf'\nabla_{{W_{left_name}}}\mathcal L={matrix_text(grad[key])}',
            'Se suman las contribuciones de los tres instantes.',code)
    for key in ('Wx','Wh','Wy'):
        old=getattr(net,key);new=old-eta*grad[key]
        add('Actualización',f'Actualizar {key}',
            rf'{key}^{{(k+1)}}={key}^{{(k)}}-\eta\nabla_{{{key}}}\mathcal L',
            rf'{key}^{{(k+1)}}={matrix_text(old)}-{fnum(eta)}{matrix_text(grad[key])}',
            rf'{key}^{{(k+1)}}={matrix_text(new)}','Descenso del gradiente manual.')
    return steps

# Capa visual adicional: conserva íntegramente los pasos matemáticos originales.
import re
import html

def _visual_context(step):
    section=step['section']; title=step['title']
    m=re.search(r'(?:Tiempo |t=)([123])',title)
    if not m: m=re.search(r'Contribución en t=([123])',title)
    t=int(m.group(1)) if m else (3 if section=='Forward' else 1)
    if step.get('code')=='B2': t=3
    if step.get('code')=='B5': t=2
    if step.get('code')=='B7': t=1
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
    is_delta = backward and step.get('code') in ('B3','B6','B8')
    is_output_derivative = backward and step.get('code') == 'B1'
    is_state_grad = backward and step.get('code') in ('B2','B5','B7')
    is_gwx = backward and (step.get('code')=='B10')
    is_gwh = backward and (step.get('code')=='B9')
    is_gwy = backward and step.get('code')=='B4'
    is_update = section == 'Actualización'
    forward_hidden = section == 'Forward' and title.startswith('Tiempo ')
    forward_output = section == 'Forward' and 'Predicción' in title
    forward_loss = section == 'Forward' and title == 'Pérdida'
    # No iluminar la salida al calcular delta_2 o delta_1.
    output_edge = forward_output or (is_delta and t == 3) or is_gwy or (is_state_grad and step.get('code')=='B2')
    loss_edge = forward_loss or is_output_derivative
    # Cada etapa ilumina solo su tramo temporal correspondiente.
    recurrent_edge = ((forward_hidden and t > 1) or
                      (is_delta and t in (1, 2)) or is_gwh or (is_state_grad and not step.get('code')=='B2'))
    input_edge = forward_hidden or is_gwx
    selected_times = ({t} if forward_hidden or is_delta else
                      {t} if ('Contribución en t=' in title) else set())
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

# Guía del algoritmo con selección interactiva por derivada.
BPTT_GUIDE = [
 ('B1',r'\frac{\partial\mathcal L}{\partial\hat y_3}=\hat y_3-y','Error de salida'),
 ('B2',r'\nabla_{h_3}\mathcal L=W_y^T(\hat y_3-y)','Llevar el error a h₃'),
 ('B3',r'\delta_3=D_3\nabla_{h_3}\mathcal L','Atravesar tanh en t=3'),
 ('B4',r'\nabla_{W_y}\mathcal L=(\hat y_3-y)h_3^T','Gradiente de salida'),
 ('B5',r'\nabla_{h_2}\mathcal L=W_h^T\delta_3','Retroceder al tiempo 2'),
 ('B6',r'\delta_2=D_2W_h^T\delta_3','Atravesar tanh en t=2'),
 ('B7',r'\nabla_{h_1}\mathcal L=W_h^T\delta_2','Retroceder al tiempo 1'),
 ('B8',r'\delta_1=D_1W_h^T\delta_2','Atravesar tanh en t=1'),
 ('B9',r'\nabla_{W_h}\mathcal L=\sum_{t=1}^3\delta_th_{t-1}^T','Acumular pesos recurrentes'),
 ('B10',r'\nabla_{W_x}\mathcal L=\sum_{t=1}^3\delta_tx_t^T','Acumular pesos de entrada'),
]

def bptt_step_info(step):
    return next((i for i,z in enumerate(BPTT_GUIDE) if z[0]==step.get('code')),None)

def show_bptt_roadmap():
    st.subheader('Ruta interactiva del Backward · B1–B10')
    st.write('Selecciona una derivada para identificar de dónde viene su gradiente y para qué se utiliza después.')
    if 'roadmap_selected' not in st.session_state: st.session_state.roadmap_selected=0
    # Diagrama ramificado: la ruta principal y los tres gradientes de parámetros.
    st.markdown('#### Diagrama de dependencias')
    st.graphviz_chart(r"""digraph BPTT {
      graph [rankdir=TB, bgcolor="transparent", nodesep=0.32, ranksep=0.36];
      node [shape=box, style="rounded,filled", fillcolor="#edf3fa", color="#8fa5bd", fontname="Helvetica", fontsize=13];
      edge [color="#7d92ab", arrowsize=.7];
      B1 [label="B1 · Error de salida"];
      B2 [label="B2 · Gradiente h₃"];
      B3 [label="B3 · δ₃"];
      B4 [label="B4 · ∇Wy", fillcolor="#e2f5f0"];
      B5 [label="B5 · Gradiente h₂"];
      B6 [label="B6 · δ₂"];
      B7 [label="B7 · Gradiente h₁"];
      B8 [label="B8 · δ₁"];
      B9 [label="B9 · ∇Wh", fillcolor="#e2f5f0"];
      B10 [label="B10 · ∇Wx", fillcolor="#e2f5f0"];
      B1 -> B2 -> B3 -> B5 -> B6 -> B7 -> B8;
      B1 -> B4;
      B3 -> B9 [style=dashed]; B6 -> B9 [style=dashed]; B8 -> B9 [style=dashed];
      B3 -> B10 [style=dashed]; B6 -> B10 [style=dashed]; B8 -> B10 [style=dashed];
    }""", use_container_width=True)
    cols=st.columns(5)
    for i,(code,formula,purpose) in enumerate(BPTT_GUIDE):
        with cols[i%5]:
            if st.button(f'{code} · {purpose}',key=f'route_{code}',use_container_width=True,
                         type='primary' if i==st.session_state.roadmap_selected else 'secondary'):
                st.session_state.roadmap_selected=i
    code,formula,purpose=BPTT_GUIDE[st.session_state.roadmap_selected]
    st.markdown(f'### {code} · {purpose}')
    st.latex(formula)
    if code in ('B3','B6','B8'):
        t={'B3':3,'B6':2,'B8':1}[code]
        D=np.diag(1-result['h'][t]**2)
        st.latex(rf'D_{{{t}}}=\frac{{\partial\mathbf h_{{{t}}}}}{{\partial\mathbf a_{{{t}}}}}=\operatorname{{diag}}(1-h_{{{t},1}}^2,1-h_{{{t},2}}^2)={matrix_text(D)}')
    if code in ('B9','B10'):
        st.caption('Los tres deltas calculados en B3, B6 y B8 alimentan este gradiente de pesos compartidos.')
    if st.button(f'Ir al cálculo de {code}',type='primary'):
        index=next((i for i,z in enumerate(steps) if z.get('code')==code),None)
        if index is not None:
            st.session_state.step=index
            st.info('Paso seleccionado. Abre la pestaña Cálculo paso a paso para verlo.')
    st.markdown('### Orden definitivo de las derivadas')
    for number,formula,purpose in BPTT_GUIDE:
        c1,c2,c3=st.columns([1,4,5]);c1.markdown(f'**{number}**');c2.latex(formula);c3.write(purpose)
    st.markdown('### Jacobiano de la activación tanh')
    st.latex(r'\mathbf h_t=\tanh(\mathbf a_t),\qquad D_t=\frac{\partial\mathbf h_t}{\partial\mathbf a_t}=\begin{pmatrix}1-h_{t,1}^2&0\\0&1-h_{t,2}^2\end{pmatrix}')
    st.markdown('### Acumulación de los gradientes')
    st.latex(r'G_h^{(t)}=\delta_t h_{t-1}^T,\qquad \nabla_{W_h}\mathcal L=\sum_{t=1}^{3}G_h^{(t)}')
    st.latex(r'G_x^{(t)}=\delta_t x_t^T,\qquad \nabla_{W_x}\mathcal L=\sum_{t=1}^{3}G_x^{(t)}')

route, manual, training=st.tabs(['Ruta del Backward', 'Cálculo paso a paso','Entrenamiento visual'])
with route:
    with st.expander('📐 Matrices de pesos actuales (se actualizan con cada época)',expanded=True):
        mcols=st.columns(3)
        for col,name in zip(mcols,('Wx','Wh','Wy')):
            with col:
                st.markdown(f'**{name} — {getattr(net,name).shape[0]} × {getattr(net,name).shape[1]}**')
                st.latex(rf'{name}={matrix_text(getattr(net,name))}')
    show_bptt_roadmap()

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
    if current['section']=='Backward':
        guide_idx=bptt_step_info(current)
        if guide_idx is not None:
            code, formula, purpose = BPTT_GUIDE[guide_idx]
            st.markdown(f'### {code} · {purpose}')
            st.caption('Fórmula de referencia del paso que estamos desarrollando:')
            st.latex(formula)
        else:
            st.caption('Cálculo matricial del Backward.')
    st.markdown(f'**{current["phase"]}**')
    st.latex(current['latex'])
    st.write(current['explanation'])
    with st.expander('Historial de pasos ya revelados',expanded=False):
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
