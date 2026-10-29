
import streamlit as st

def mostrar_werbos_sensibilidades():
    st.header("Sensibilidad, propagación y acumulación de gradientes")
    st.caption("Werbos (1990) ↔ nuestra RNN 2–2–1, T = 3")

    st.info(
        "Objetivo: comprender qué calcula cada operación del "
        "pseudocódigo de Werbos y cómo se relaciona con BPTT."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Sensibilidad",
        "2. Propagación",
        "3. Acumulación",
        "4. Comparación"
    ])

    # ======================================================
    # 1. SENSIBILIDAD
    # ======================================================
    with tab1:
        st.subheader("¿Qué es una sensibilidad?")

        st.write(
            "Una sensibilidad es una derivada de la función "
            "de pérdida respecto de una variable de la red."
        )

        st.markdown("**Notación de Werbos:**")
        st.latex(r"F_{x_i(t)}")

        st.markdown("**Nuestra notación:**")
        st.latex(
            r"F_{x_i(t)}"
            r"\longleftrightarrow"
            r"\frac{\partial\mathcal L}{\partial h_{t,i}}"
        )

        st.latex(
            r"F_{\operatorname{net}_i(t)}"
            r"\longleftrightarrow"
            r"\delta_{t,i}"
            r"=\frac{\partial\mathcal L}{\partial a_{t,i}}"
        )

        st.markdown("**Ejemplo numérico:**")
        st.latex(
            r"\boxed{\delta_{3,1}"
            r"=\frac{\partial\mathcal L}{\partial a_{3,1}}"
            r"=-0.128432}"
        )

        st.success(
            "Interpretación: esta derivada mide la sensibilidad "
            "de la pérdida respecto de la preactivación "
            "de la primera neurona en t = 3."
        )

        with st.expander("¿Qué diferencia hay entre h y a?"):
            st.latex(r"a_t=W_xx_t+W_hh_{t-1}")
            st.latex(r"h_t=\tanh(a_t)")
            st.write(
                "a es la preactivación y h es la activación. "
                "Sus sensibilidades no son iguales: se relacionan "
                "mediante la derivada de tanh."
            )
            st.latex(
                r"\delta_t=D_t\nabla_{h_t}\mathcal L"
            )

    # ======================================================
    # 2. PROPAGACIÓN
    # ======================================================
    with tab2:
        st.subheader("¿Qué significa propagar sensibilidad?")

        st.write(
            "Significa utilizar una sensibilidad posterior "
            "que ya conocemos para calcular una sensibilidad anterior."
        )

        st.markdown("**Werbos:**")
        st.latex(
            r"F_{x_i(t)}\leftarrow F_{x_i(t)}"
            r"+W'_{ji}F_{\operatorname{net}_j(t+1)}"
        )

        st.markdown("**Nuestra RNN — B5:**")
        st.latex(
            r"\boxed{\nabla_{h_2}\mathcal L=W_h^T\delta_3}"
        )

        st.markdown("**¿De dónde sale el peso?**")
        st.latex(
            r"a_{t+1,j}=\cdots+(W_h)_{ji}h_{t,i}+\cdots"
        )
        st.latex(
            r"\frac{\partial a_{t+1,j}}{\partial h_{t,i}}"
            r"=(W_h)_{ji}"
        )

        st.write(
            "Por la regla de la cadena, cada neurona posterior "
            "aporta una contribución a la derivada anterior."
        )

        st.latex(
            r"\frac{\partial\mathcal L}{\partial h_{2,i}}"
            r"=\sum_{j=1}^{2}(W_h)_{ji}\delta_{3,j}"
        )

        st.markdown("**Ejemplo con nuestros pesos:**")
        st.latex(
            r"W_h=\begin{pmatrix}"
            r"0.5&0.1\\0.1&0.5"
            r"\end{pmatrix}"
        )
        st.latex(
            r"\delta_3=\begin{pmatrix}"
            r"-0.128432\\-0.189010"
            r"\end{pmatrix}"
        )

        st.markdown("**Primera neurona de t = 2:**")
        st.latex(
            r"\frac{\partial\mathcal L}{\partial h_{2,1}}"
            r"=0.5(-0.128432)+0.1(-0.189010)"
        )
        st.latex(r"\boxed{-0.083117}")

        st.markdown("**¿Qué significa actualizar?**")
        st.write(
            "Actualizar significa sumar una nueva contribución "
            "a la derivada que estamos construyendo."
        )

        st.latex(r"S_0=0")
        st.latex(r"S_1=0+0.5(-0.128432)=-0.064216")
        st.latex(
            r"S_2=-0.064216+0.1(-0.189010)=-0.083117"
        )

        st.success(
            "Propagar: calcular una derivada anterior. "
            "Actualizar: añadir una contribución al valor acumulado."
        )

        with st.expander("Relación con B6"):
            st.write(
                "B5 calcula la sensibilidad respecto de h₂. "
                "B6 la transforma en sensibilidad respecto de a₂."
            )
            st.latex(
                r"\boxed{\delta_2=D_2W_h^T\delta_3}"
            )

    # ======================================================
    # 3. ACUMULACIÓN DE GRADIENTES
    # ======================================================
    with tab3:
        st.subheader("¿Qué significa acumular gradientes?")

        st.write(
            "Es sumar las contribuciones a la derivada "
            "de un mismo peso compartido."
        )

        st.markdown("**Werbos:**")
        st.latex(
            r"F_{W'_{ij}}\leftarrow F_{W'_{ij}}"
            r"+F_{\operatorname{net}_i(t)}x_j(t-1)"
        )

        st.markdown("**Nuestra RNN — B9:**")
        st.latex(
            r"\boxed{\nabla_{W_h}\mathcal L"
            r"=\sum_{t=1}^{3}\delta_th_{t-1}^T}"
        )

        st.latex(
            r"\nabla_{W_h}\mathcal L"
            r"=\delta_1h_0^T+\delta_2h_1^T+\delta_3h_2^T"
        )

        st.markdown("**¿Por qué se suman?**")
        st.write(
            "Porque la misma matriz Wₕ participa en "
            "los tres instantes. Cada uso aporta una parte "
            "de la derivada respecto de esa matriz."
        )

        st.latex(
            r"\underbrace{\delta_1h_0^T}_{t=1}"
            r"+\underbrace{\delta_2h_1^T}_{t=2}"
            r"+\underbrace{\delta_3h_2^T}_{t=3}"
        )

        st.info(
            "Como h₀ = 0 en nuestro ejercicio, "
            "la primera contribución al gradiente de Wₕ es cero."
        )

        st.markdown("**Resultado numérico B9:**")
        st.latex(
            r"\nabla_{W_h}\mathcal L="
            r"\begin{pmatrix}"
            r"-0.042127&-0.041740\\"
            r"-0.058595&-0.059710"
            r"\end{pmatrix}"
        )

        st.success(
            "Aquí no estamos calculando una sensibilidad anterior. "
            "Estamos reuniendo las contribuciones a los pesos recurrentes."
        )

    # ======================================================
    # 4. COMPARACIÓN
    # ======================================================
    with tab4:
        st.subheader("Las tres operaciones, una al lado de otra")

        st.markdown("### Sensibilidad")
        st.latex(
            r"F_{\operatorname{net}_i(t)}"
            r"\longleftrightarrow\delta_{t,i}"
        )
        st.write("Es una derivada de la pérdida.")

        st.divider()

        st.markdown("### Propagación — B5")
        st.latex(
            r"F_{x_i(t)}\leftarrow F_{x_i(t)}"
            r"+W'_{ji}F_{\operatorname{net}_j(t+1)}"
        )
        st.latex(
            r"\nabla_{h_2}\mathcal L=W_h^T\delta_3"
        )
        st.write(
            "Calcula sensibilidades anteriores "
            "a partir de sensibilidades posteriores."
        )

        st.divider()

        st.markdown("### Acumulación — B9")
        st.latex(
            r"F_{W'_{ij}}\leftarrow F_{W'_{ij}}"
            r"+F_{\operatorname{net}_i(t)}x_j(t-1)"
        )
        st.latex(
            r"\nabla_{W_h}\mathcal L"
            r"=\sum_{t=1}^{3}\delta_th_{t-1}^T"
        )
        st.write(
            "Suma las contribuciones a los pesos "
            "compartidos a través del tiempo."
        )

        st.divider()

        st.markdown("### La diferencia fundamental")
        st.latex(
            r"\underbrace{W_h^T\delta_3}_{\mathrm{B5:\ propagacion}}"
            r"\neq"
            r"\underbrace{\sum_{t=1}^{3}\delta_th_{t-1}^T}"
            r"_{\mathrm{B9:\ acumulacion}}"
        )

   
