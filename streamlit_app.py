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
    h=result['h'][-1]
    add('Forward','Predicción final',r'\hat y_3=w^y_{11}h_{1,3}+w^y_{12}h_{2,3}',
        rf'\hat y_3={fnum(net.Wy[0,0])}({fnum(h[0])})+{fnum(net.Wy[0,1])}({fnum(h[1])})',
        rf'\hat y_3={fnum(result["yhat"])}','La salida es lineal y se calcula únicamente en el instante final.')
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
    section,t,neuron=_visual_context(step)
    backward=section=='Backward'; updating=section=='Actualización'
    active='#e59b3c' if backward else '#25a9a0'
    muted='#81909f'
    items=['<svg viewBox="0 0 840 245" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="RNN desplegada en tres tiempos">',
           '<defs><marker id="arrowforward" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#25a9a0"/></marker><marker id="arrowback" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#e59b3c"/></marker></defs>']
    for k in range(1,4):
        cx=130+(k-1)*270
        selected=k==t or updating
        stroke=active if selected else muted
        opacity='1' if selected else '.35'
        items.append(f'<rect x="{cx-107}" y="20" width="214" height="196" rx="13" fill="{active}" fill-opacity="{.075 if selected else .015}" stroke="{stroke}" stroke-opacity="{opacity}" stroke-width="{2.4 if selected else 1}"/>')
        items.append(f'<text x="{cx}" y="43" text-anchor="middle" fill="#dbe6ee" font-size="15">Instante t = {k}</text>')
        for j,y in enumerate((95,157),start=1):
            items.append(f'<circle cx="{cx-68}" cy="{y}" r="19" fill="#475569" fill-opacity=".25" stroke="{muted}"/>')
            items.append(f'<text x="{cx-68}" y="{y+5}" text-anchor="middle" fill="#e6edf4" font-size="14">x{j}</text>')
            hi=selected and (neuron is None or neuron==j)
            hc=active if hi else muted
            items.append(f'<circle cx="{cx+67}" cy="{y}" r="22" fill="{hc}" fill-opacity="{.35 if hi else .1}" stroke="{hc}" stroke-width="{3 if hi else 1}"/>')
            items.append(f'<text x="{cx+67}" y="{y+5}" text-anchor="middle" fill="#f0f6fa" font-size="14">h{j}</text>')
        for y1 in (95,157):
            for j,y2 in enumerate((95,157),start=1):
                hi=selected and (neuron is None or neuron==j)
                items.append(f'<line x1="{cx-48}" y1="{y1}" x2="{cx+44}" y2="{y2}" stroke="{active if hi else muted}" stroke-width="{2.6 if hi else 1}" stroke-opacity="{.95 if hi else .23}"/>')
        items.append(f'<text x="{cx}" y="198" text-anchor="middle" fill="#b8c8d5" font-size="12">Wₓ · activación tanh</text>')
    for k in (1,2):
        left=130+(k-1)*270+91;right=130+k*270-91
        relevant=updating or (k+1==t if not backward else k==t)
        col=active if relevant else muted
        x1,x2=(right,left) if backward else (left,right)
        marker='arrowback' if backward else 'arrowforward'
        items.append(f'<line x1="{x1}" y1="126" x2="{x2}" y2="126" stroke="{col}" stroke-width="{3.4 if relevant else 1.5}" stroke-opacity="{1 if relevant else .3}" marker-end="url(#{marker})"/>')
        items.append(f'<text x="{(left+right)/2}" y="110" text-anchor="middle" fill="{col}" font-size="13">{"Wₕᵀ" if backward else "Wₕ"}</text>')
    items.append(f'<text x="420" y="237" text-anchor="middle" fill="#dbe6ee" font-size="13">{"Retropropagación: t=3 → t=1" if backward else "Actualización de los pesos compartidos" if updating else "Propagación hacia adelante: t=1 → t=3"}</text>')
    items.append('</svg>')
    return ''.join(items)

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
    current=steps[st.session_state.step]
    show_visual_step(current,result)
    st.markdown(f'### {current["section"]} · {current["title"]}')
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
