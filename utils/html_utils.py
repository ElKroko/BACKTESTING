"""
Utilidades para la generación de HTML y elementos de interfaz de usuario.

Este módulo contiene funciones para generar elementos HTML consistentes
como tooltips, tarjetas y otros componentes de interfaz de usuario.
"""
import streamlit as st
import os

def tooltip(text, explanation):
    """
    Crea un tooltip con una explicación que aparece al pasar el mouse.
    
    Args:
        text: El texto visible que contiene el tooltip
        explanation: La explicación que aparece al pasar el ratón
        
    Returns:
        Código HTML del tooltip
    """
    tooltip_id = text.lower().replace(" ", "_").replace("&", "").replace(":", "") + "_tip"
    
    return f"""
    <div style="display: inline-flex; align-items: center; gap: 0.4rem;">
        <span>{text}</span>
        <span class="tooltip" id="{tooltip_id}">
            ℹ️
            <span class="tooltiptext">{explanation}</span>
        </span>
    </div>
    """
    
def create_info_card(title, content, icon="ℹ️", tag=None):
    """
    Crea una tarjeta de información con título, contenido e icono.
    
    Args:
        title: Título de la tarjeta
        content: Contenido principal (admite HTML)
        icon: Emoji o código HTML del icono
        tag: Etiqueta opcional (ej. "Pro", "New")
        
    Returns:
        Código HTML de la tarjeta
    """
    tag_html = f'<span class="info-card-tag">{tag}</span>' if tag else ''
    
    return f"""
    <div class="info-card">
        <div class="info-card-header">
            <span class="info-card-icon">{icon}</span>
            <h3 class="info-card-title">{title}</h3>
            {tag_html}
        </div>
        <div class="info-card-content">
            {content}
        </div>
    </div>
    """

def create_stats_row(stats_dict):
    """
    Crea una fila de estadísticas con valores y etiquetas.
    
    Args:
        stats_dict: Diccionario de estadísticas {etiqueta: valor}
        
    Returns:
        Código HTML de la fila de estadísticas
    """
    stat_items = ""
    for label, value in stats_dict.items():
        stat_items += f"""
        <div class="stat-item">
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>
        """
        
    return f"""
    <div class="stats-row">
        {stat_items}
    </div>
    """

def create_tabbed_content(tabs_dict):
    """
    Crea un contenedor con pestañas para organizar contenido.
    
    Args:
        tabs_dict: Diccionario de pestañas {nombre_pestaña: contenido_html}
        
    Returns:
        Código HTML del contenedor con pestañas
    """
    tab_buttons = ""
    tab_contents = ""
    
    for i, (tab_name, content) in enumerate(tabs_dict.items()):
        active = "active" if i == 0 else ""
        tab_id = f"tab-{tab_name.lower().replace(' ', '-')}"
        
        tab_buttons += f"""
        <button class="tab-button {active}" onclick="showTab('{tab_id}')">{tab_name}</button>
        """
        
        tab_contents += f"""
        <div id="{tab_id}" class="tab-content {active}">
            {content}
        </div>
        """
    
    # El script JavaScript para la funcionalidad de las pestañas
    js_script = """
    <script>
    function showTab(tabId) {
        // Ocultar todas las pestañas
        var contents = document.getElementsByClassName('tab-content');
        for (var i = 0; i < contents.length; i++) {
            contents[i].classList.remove('active');
        }
        
        // Desactivar todos los botones
        var buttons = document.getElementsByClassName('tab-button');
        for (var i = 0; i < buttons.length; i++) {
            buttons[i].classList.remove('active');
        }
        
        // Mostrar la pestaña seleccionada
        document.getElementById(tabId).classList.add('active');
        
        // Activar el botón correspondiente
        var activeButtons = document.querySelectorAll('.tab-button[onclick*="' + tabId + '"]');
        for (var i = 0; i < activeButtons.length; i++) {
            activeButtons[i].classList.add('active');
        }
    }
    </script>
    """
    
    return f"""
    <div class="tabbed-container">
        <div class="tab-buttons">
            {tab_buttons}
        </div>
        <div class="tab-contents">
            {tab_contents}
        </div>
    </div>
    {js_script}
    """

def apply_tooltip_css():
    """
    Aplica los estilos CSS necesarios para los tooltips.
    Lee el archivo CSS desde static/css/tooltips.css.
    """
    css_path = os.path.join('static', 'css', 'tooltips.css')
    
    # Verificar si el archivo existe
    if os.path.exists(css_path):
        with open(css_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
    else:
        # CSS predeterminado si el archivo no existe - usando las mismas clases que el tooltip()
        css_content = """
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        
        .tooltiptext {
            visibility: hidden;
            position: absolute;
            z-index: 100;
            width: 300px;
            bottom: 125%;
            left: 50%;
            margin-left: -150px;
            opacity: 0;
            transition: opacity 0.3s;
            background-color: rgba(50, 50, 50, 0.95);
            color: #fff;
            border-radius: 6px;
            padding: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }
        
        .tooltip::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -8px;
            width: 0;
            height: 0;
            border-left: 8px solid transparent;
            border-right: 8px solid transparent;
            border-top: 8px solid rgba(50, 50, 50, 0.95);
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        """
    
    # Aplicar CSS
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

def apply_theme_css(palette):
    """
    Aplica los estilos CSS de temas basados en la paleta seleccionada.
    Lee el archivo CSS desde static/css/theme.css y lo personaliza con la paleta.
    
    Args:
        palette: Diccionario con colores de la paleta actual
    """
    css_path = os.path.join('static', 'css', 'theme.css')
    
    # Verificar si el archivo existe
    if os.path.exists(css_path):
        with open(css_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
    else:
        # CSS predeterminado si el archivo no existe
        css_content = """
        /* Estilos de tema personalizados */
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: var(--secondary-color);
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: var(--secondary-color);
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--accent-color);
            color: var(--text-color);
        }
        
        .stMarkdown a {
            color: var(--accent-color);
        }
        
        .stDataFrame {
            border: 1px solid var(--border-color);
        }
        
        /* Estilo personalizado para formularios */
        .stForm {
            background-color: var(--secondary-color);
            padding: 20px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }
        
        /* Tarjetas de información */
        .info-card {
            background-color: var(--secondary-color);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            border: 1px solid var(--border-color);
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .info-card-header {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .info-card-icon {
            font-size: 24px;
            margin-right: 10px;
        }
        
        .info-card-title {
            margin: 0;
            font-size: 18px;
            font-weight: 600;
            color: var(--text-color);
        }
        
        .info-card-tag {
            background-color: var(--accent-color);
            color: white;
            border-radius: 12px;
            padding: 2px 8px;
            font-size: 12px;
            margin-left: 10px;
        }
        
        /* Fila de estadísticas */
        .stats-row {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat-item {
            background-color: var(--secondary-color);
            border-radius: 8px;
            padding: 12px;
            min-width: 120px;
            text-align: center;
            border: 1px solid var(--border-color);
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: var(--accent-color);
        }
        
        .stat-label {
            font-size: 14px;
            color: var(--text-color);
            opacity: 0.8;
        }
        """
    
    # Personalizar CSS con variables de la paleta
    custom_css = f"""
    <style>
    :root {{
        --background-color: {palette['background']};
        --text-color: {palette['text']};
        --accent-color: {palette['accent']};
        --secondary-color: {palette['secondary']};
        --border-color: {palette['border']};
    }}
    
    {css_content}
    </style>
    """
    
    # Aplicar CSS
    st.markdown(custom_css, unsafe_allow_html=True)

def apply_backtest_summary_css():
    """
    Aplica los estilos CSS para la visualización del resumen de backtest.
    Lee el archivo CSS desde static/css/backtest_summary.css.
    """
    css_path = os.path.join('static', 'css', 'backtest_summary.css')
    
    # Verificar si el archivo existe
    if os.path.exists(css_path):
        with open(css_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
    else:
        # CSS predeterminado si el archivo no existe
        css_content = """
        /* Estilos para el resumen de backtest */
        .backtest-summary {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .summary-item {
            display: flex;
            align-items: center;
            margin: 5px 10px;
        }
        
        .summary-icon {
            font-size: 1.2rem;
            margin-right: 8px;
            width: 24px;
            text-align: center;
        }
        
        .summary-label {
            font-size: 0.8rem;
            opacity: 0.8;
            margin-right: 4px;
        }
        
        .summary-value {
            font-weight: bold;
            font-size: 0.9rem;
        }
        """
    
    # Aplicar CSS
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)