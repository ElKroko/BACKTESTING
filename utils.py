# utils.py
import streamlit as st

# --- Utility function for tooltips ---
def tooltip(label, explanation, icon="ℹ️"):
    """
    Creates a styled tooltip for educational purposes.
    
    Args:
        label: The text to show (title/header)
        explanation: The explanation to show in the tooltip
        icon: The icon to use (default: info icon)
    
    Returns:
        HTML for the tooltip
    """
    tooltip_id = label.lower().replace(" ", "_").replace("&", "").replace(":", "") + "_tip"
    
    return f"""
    <div style="display: inline-flex; align-items: center; gap: 0.4rem;">
        <span>{label}</span>
        <span class="tooltip" id="{tooltip_id}">
            {icon}
            <span class="tooltiptext">{explanation}</span>
        </span>
    </div>
    """

# --- CSS for tooltips ---
def apply_tooltip_css():
    st.markdown("""
    <style>
    /* Tooltip container */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        font-size: 0.9rem;
    }

    /* Tooltip text */
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 280px;
        background-color: rgba(50, 50, 50, 0.98);
        color: #fff;
        text-align: left;
        padding: 8px 12px;
        border-radius: 6px;
        border: 1px solid #666;
        font-size: 0.85rem;
        line-height: 1.4;
        
        /* Position the tooltip text */
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        margin-left: -140px;
        
        /* Fade in tooltip */
        opacity: 0;
        transition: opacity 0.3s;
    }

    /* Tooltip arrow */
    .tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #666 transparent transparent transparent;
    }

    /* Show the tooltip text when you mouse over the tooltip container */
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)