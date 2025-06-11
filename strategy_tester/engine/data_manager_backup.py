"""
Gestor de datos - Descarga y maneja datos de diferentes fuentes
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle
import json
from typing import Optional, Dict, List, Tuple
import hashlib


class DataManager:
    """
    Gestor de datos inteligente para descargar, almacenar y recuperar datos de mercado
    con sistema de cache avanzado
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        self.metadata_dir = os.path.join(cache_dir, "metadata")
        self.exchange = None
        self._ensure_cache_dirs()
          def _ensure_cache_dirs(self):
        """Crear directorios de cache si no existen"""
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
    def _get_cache_filename(self, symbol: str, timeframe: str, start_date: str) -> str:
        """Generar nombre de archivo para cache"""
        clean_symbol = symbol.replace('/', '_')
        clean_start = start_date.replace(':', '-').replace('T', '_').replace('Z', '')
        return f"{clean_symbol}_{timeframe}_{clean_start}.pkl"
    
    def _get_metadata_filename(self, symbol: str, timeframe: str, start_date: str) -> str:
        """Generar nombre de archivo para metadatos"""
        clean_symbol = symbol.replace('/', '_')
        clean_start = start_date.replace(':', '-').replace('T', '_').replace('Z', '')
        return f"{clean_symbol}_{timeframe}_{clean_start}_meta.json"
    
    def _get_cache_key(self, symbol: str, timeframe: str, start_date: str) -> str:
        """Generar clave única para identificar el cache"""
        key_string = f"{symbol}_{timeframe}_{start_date}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _save_metadata(self, symbol: str, timeframe: str, start_date: str, 
                      df: pd.DataFrame, end_date: Optional[str] = None):
        """Guardar metadatos del cache"""
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date or datetime.now().isoformat(),
            'actual_start': df.index[0].isoformat() if len(df) > 0 else None,
            'actual_end': df.index[-1].isoformat() if len(df) > 0 else None,
            'total_candles': len(df),
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'cache_key': self._get_cache_key(symbol, timeframe, start_date),
            'file_size_mb': 0  # Se calculará después
        }
        
        meta_file = os.path.join(self.metadata_dir, 
                                self._get_metadata_filename(symbol, timeframe, start_date))
        
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self, symbol: str, timeframe: str, start_date: str) -> Optional[Dict]:
        """Cargar metadatos del cache"""
        meta_file = os.path.join(self.metadata_dir, 
                                self._get_metadata_filename(symbol, timeframe, start_date))
        
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                return json.load(f)
        return None
    
    def _is_cache_valid(self, symbol: str, timeframe: str, start_date: str, 
                       max_age_hours: int = 1) -> Tuple[bool, Optional[Dict]]:
        """
        Verificar si el cache es válido
        
        Args:
            symbol: Símbolo
            timeframe: Marco temporal
            start_date: Fecha de inicio
            max_age_hours: Máximo tiempo en horas para considerar el cache válido
            
        Returns:
            (es_válido, metadatos)
        """
        cache_file = os.path.join(self.cache_dir, 
                                 self._get_cache_filename(symbol, timeframe, start_date))
        
        if not os.path.exists(cache_file):
            return False, None
        
        metadata = self._load_metadata(symbol, timeframe, start_date)
        if not metadata:
            return False, None
        
        # Verificar edad del cache
        last_updated = datetime.fromisoformat(metadata['last_updated'])
        age_hours = (datetime.now() - last_updated).total_seconds() / 3600
        
        if age_hours > max_age_hours:
            print(f"🕐 Cache obsoleto: {age_hours:.1f}h > {max_age_hours}h")
            return False, metadata
        
        # Verificar integridad del archivo
        try:
            with open(cache_file, 'rb') as f:
                pickle.load(f)
            return True, metadata
        except Exception as e:
            print(f"⚠️ Cache corrupto: {e}")
            return False, metadata
    
    def _needs_update(self, symbol: str, timeframe: str, start_date: str, 
                     end_date: Optional[str] = None) -> bool:
        """
        Determinar si necesita actualizar datos existentes
        """
        metadata = self._load_metadata(symbol, timeframe, start_date)
        if not metadata:
            return True
        
        # Si no hay end_date, siempre necesita actualización para datos recientes
        if not end_date:
            now = datetime.now()
            last_data = datetime.fromisoformat(metadata['actual_end'])
            
            # Calcular cuánto tiempo debería haber entre actualizaciones
            if timeframe == '1m':
                update_threshold = timedelta(minutes=30)
            elif timeframe == '5m':
                update_threshold = timedelta(hours=1)
            elif timeframe == '1h':
                update_threshold = timedelta(hours=6)
            elif timeframe == '4h':
                update_threshold = timedelta(hours=12)
            elif timeframe == '1d':
                update_threshold = timedelta(days=1)
            else:
                update_threshold = timedelta(hours=2)
            
            return (now - last_data) > update_threshold
        
        return False
    
    def _setup_exchange(self, exchange_name: str = 'binance'):
        """Configurar exchange para descargas"""
        if self.exchange is None:
            if exchange_name.lower() == 'binance':
                self.exchange = ccxt.binance()
            elif exchange_name.lower() == 'bybit':
                self.exchange = ccxt.bybit()
            else:
                raise ValueError(f"Exchange {exchange_name} no soportado")
    
    def download_data(self, 
                     symbol: str = 'BTC/USDT', 
                     timeframe: str = '1h',
                     start_date: str = '2023-01-01T00:00:00Z',
                     end_date: Optional[str] = None,
                     exchange: str = 'binance',
                     use_cache: bool = True) -> pd.DataFrame:
        """
        Descargar datos de mercado
        
        Args:
            symbol: Par de trading (ej: 'BTC/USDT')
            timeframe: Marco temporal ('1m', '5m', '15m', '1h', '4h', '1d')
            start_date: Fecha de inicio en formato ISO
            end_date: Fecha de fin (opcional, por defecto hasta ahora)
            exchange: Exchange a usar ('binance', 'bybit')
            use_cache: Usar cache local si está disponible
            
        Returns:
            DataFrame con datos OHLCV
        """
        # Verificar cache primero
        if use_cache:
            cache_file = os.path.join(self.cache_dir, self._get_cache_filename(symbol, timeframe, start_date))
            if os.path.exists(cache_file):
                print(f"📁 Cargando datos desde cache: {symbol} ({timeframe})")
                with open(cache_file, 'rb') as f:
                    df = pickle.load(f)
                    print(f"✅ Datos cargados: {len(df)} velas desde {df.index[0]} hasta {df.index[-1]}")
                    return df
        
        # Descargar datos frescos
        self._setup_exchange(exchange)
        
        print(f"🌐 Descargando datos de {exchange.upper()}: {symbol} ({timeframe})")
        
        try:
            # Convertir fecha de inicio
            since = self.exchange.parse8601(start_date)
            
            # Descargar datos
            all_ohlcv = []
            current_since = since
            
            # Calcular límites de tiempo
            timeframe_seconds = self.exchange.parse_timeframe(timeframe) * 1000
            now = self.exchange.milliseconds()
            
            if end_date:
                end_timestamp = self.exchange.parse8601(end_date)
            else:
                end_timestamp = now
            
            while current_since < end_timestamp:
                try:
                    # Descargar lote de datos
                    ohlcv_batch = self.exchange.fetch_ohlcv(
                        symbol, 
                        timeframe=timeframe, 
                        since=current_since,
                        limit=1000  # Máximo típico por request
                    )
                    
                    if not ohlcv_batch:
                        break
                        
                    all_ohlcv.extend(ohlcv_batch)
                    
                    # Actualizar timestamp para siguiente lote
                    current_since = ohlcv_batch[-1][0] + timeframe_seconds
                    
                    # Pausa para evitar rate limiting
                    self.exchange.sleep(100)  # 100ms
                    
                    print(f"  📊 Descargados: {len(all_ohlcv)} velas...", end='\r')
                    
                except Exception as e:
                    print(f"\n  ⚠️ Error en lote: {e}")
                    break
            
            if not all_ohlcv:
                raise ValueError("No se pudieron descargar datos")
            
            # Convertir a DataFrame
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Procesar timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Remover duplicados y ordenar
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)
            
            # Renombrar columnas para consistencia
            df.columns = ['open', 'high', 'low', 'close', 'vol']
            
            print(f"\n✅ Datos descargados: {len(df)} velas desde {df.index[0]} hasta {df.index[-1]}")
            
            # Guardar en cache
            if use_cache:
                cache_file = os.path.join(self.cache_dir, self._get_cache_filename(symbol, timeframe, start_date))
                with open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
                print(f"💾 Datos guardados en cache")
            
            return df
            
        except Exception as e:
            print(f"❌ Error descargando datos: {e}")
            raise
    
    def get_multiple_symbols(self, 
                           symbols: List[str],
                           timeframe: str = '1h',
                           start_date: str = '2023-01-01T00:00:00Z',
                           exchange: str = 'binance') -> Dict[str, pd.DataFrame]:
        """
        Descargar datos para múltiples símbolos
        
        Args:
            symbols: Lista de símbolos a descargar
            timeframe: Marco temporal
            start_date: Fecha de inicio
            exchange: Exchange a usar
            
        Returns:
            Diccionario con DataFrames por símbolo
        """
        results = {}
        
        for symbol in symbols:
            try:
                print(f"\n🔄 Descargando {symbol}...")
                df = self.download_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    exchange=exchange
                )
                results[symbol] = df
                
            except Exception as e:
                print(f"❌ Error con {symbol}: {e}")
                continue
        
        return results
    
    def clear_cache(self):
        """Limpiar cache de datos"""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            self._ensure_cache_dir()
            print("🗑️ Cache limpiado")
    
    def list_cached_data(self) -> List[str]:
        """Listar datos en cache"""
        if not os.path.exists(self.cache_dir):
            return []
        
        cache_files = []
        for file in os.listdir(self.cache_dir):
            if file.endswith('.pkl'):
                cache_files.append(file)
        
        return cache_files
    
    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """
        Obtener información sobre un dataset
        
        Args:
            df: DataFrame con datos
            
        Returns:
            Diccionario con información del dataset
        """
        return {
            'total_candles': len(df),
            'start_date': df.index[0] if len(df) > 0 else None,
            'end_date': df.index[-1] if len(df) > 0 else None,
            'missing_data': df.isnull().sum().sum(),
            'columns': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'price_range': {
                'min_price': df['low'].min() if 'low' in df.columns else None,
                'max_price': df['high'].max() if 'high' in df.columns else None,
                'avg_volume': df['vol'].mean() if 'vol' in df.columns else None
            }
        }


# Función de conveniencia para uso rápido
def quick_download(symbol: str = 'BTC/USDT', 
                  timeframe: str = '1h', 
                  days_back: int = 365) -> pd.DataFrame:
    """
    Descarga rápida de datos para los últimos N días
    
    Args:
        symbol: Símbolo a descargar
        timeframe: Marco temporal
        days_back: Días hacia atrás desde hoy
        
    Returns:
        DataFrame con datos
    """
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%dT00:00:00Z')
    
    data_manager = DataManager()
    return data_manager.download_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date
    )
