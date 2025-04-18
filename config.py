# config.py

# Default palette key
DEFAULT_PALETTE = 'Default'

# Definition of multiple named palettes
PALETTES = {
    'Default': {
        # Plotly chart colors
        'template': 'plotly_dark',
        'equity': '#00CC96',    # Jade green
        'entries': '#AB63FA',   # Purple
        'exits': '#EF553B',     # Coral red
        'chart_bg': '#1E1E1E',  # Dark gray for chart background
        'chart_grid': '#333333',# Slightly lighter gray for chart grid
        'chart_text': '#FFFFFF',# White for chart text
        
        # App UI colors
        'background': '#0E1117',  # Dark background
        'text': '#FAFAFA',        # Light text
        'accent': '#EF553B',      # Coral red (for active tabs, buttons)
        'secondary': '#262730',   # Dark gray (for card backgrounds)
        'border': '#4B5D78',      # Blue-gray for borders
        'metric_text': '#FFFFFF', # Light text for metrics
        'metric_value': '#00CC96',# Green for metric values
        'metric_bg': '#262730',   # Dark gray for metric background
        'table_bg': '#262730',    # Dark gray for table background
        'table_text': '#FFFFFF',  # White for table text
        'table_header_bg': '#333333',  # Slightly lighter gray for table headers
    },
    'Light': {
        # Plotly chart colors
        'template': 'plotly_white',
        'equity': '#636EFA',    # Blue
        'entries': '#00CC96',   # Jade green
        'exits': '#EF553B',     # Coral red
        'chart_bg': '#FFFFFF',  # White for chart background
        'chart_grid': '#EEEEEE',# Light gray for chart grid
        'chart_text': '#111111',# Dark text for charts
        
        # App UI colors
        'background': '#FFFFFF',  # White background
        'text': '#111111',        # Dark text
        'accent': '#636EFA',      # Blue (for active tabs, buttons)
        'secondary': '#F0F2F6',   # Light gray (for card backgrounds)
        'border': '#CCCCCC',      # Light gray for borders
        'metric_text': '#111111', # Dark text for metrics
        'metric_value': '#636EFA',# Blue for metric values
        'metric_bg': '#F0F2F6',   # Light gray for metric background
        'table_bg': '#FFFFFF',    # White for table background
        'table_text': '#111111',  # Dark for table text
        'table_header_bg': '#F0F2F6',  # Light gray for table headers
    },
    'Seaborn': {
        # Plotly chart colors
        'template': 'seaborn',
        'equity': '#2ca02c',    # Green
        'entries': '#d62728',   # Red
        'exits': '#1f77b4',     # Blue
        'chart_bg': '#F0F4F8',  # Light blue-gray for chart background
        'chart_grid': '#E1E8ED',# Lighter blue-gray for chart grid
        'chart_text': '#212529',# Dark text for charts
        
        # App UI colors
        'background': '#F0F4F8',  # Light blue-gray background
        'text': '#212529',        # Dark text
        'accent': '#1f77b4',      # Blue (for active tabs, buttons)
        'secondary': '#E1E8ED',   # Light blue-gray (for card backgrounds)
        'border': '#97A6B5',      # Medium blue-gray for borders
        'metric_text': '#212529', # Dark text for metrics
        'metric_value': '#1f77b4',# Blue for metric values
        'metric_bg': '#E1E8ED',   # Light blue-gray for metric background
        'table_bg': '#F0F4F8',    # Light blue-gray for table background
        'table_text': '#212529',  # Dark for table text
        'table_header_bg': '#E1E8ED',  # Light blue-gray for table headers
    }
}

# Current palette used by the application
PALETTE = PALETTES[DEFAULT_PALETTE]
