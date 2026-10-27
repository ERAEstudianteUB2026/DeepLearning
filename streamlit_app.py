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
