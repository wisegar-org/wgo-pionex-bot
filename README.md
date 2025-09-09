# Pionex CE+ADX Bot (PERP)

Sigue el README anterior que te compartí.
Te lo cuento en modo súper simple, paso a paso 😊

¿Qué mira el bot?

Mercado: BTC_USDT_PERP (futuros de Pionex).

Temporalidad por defecto: 15 minutos.

Indicadores:

CE (Chandelier Exit) con parámetros (22, 2.0). Dibuja dos “líneas de stop” que siguen el precio.

ADX(14) mide si hay fuerza/tendencia (cuanto más alto, más tendencia).

(Opcional) EMA200 como filtro de tendencia.

IA cazadora de patrones: mira el dibujo de las últimas 64 velas (retornos, RSI y ADX) y decide si ese patrón históricamente funciona o no.

¿Cuándo entra al mercado?

Señal técnica (sin IA):

Long (comprar): el precio cruza por encima de la línea CE de cortos y el ADX > 35.

Short (vender): el precio cruza por debajo de la línea CE de largos y el ADX > 35.

(Si activas EMA200) solo toma long si el precio está sobre la EMA200 y short si está bajo.

Filtro de IA (si está activado):

La IA convierte las últimas 64 velas en un vector (retornos, RSI, ADX), lo compara con K=20 “patrones-tipo” (K-means) y pregunta:

“¿Este patrón, históricamente, gana lo suficiente?”

Calcula la probabilidad de acierto del patrón y acepta la entrada solo si supera un mínimo acorde a tu TP/SL.

Con TP=2.5×ATR y SL=1.5×ATR, la probabilidad base mínima ronda ≈ 37.5%.
El bot le suma un pequeño margen (≈ +3%), así que pide ≈ 40.5% o más.

¿Dónde pone los stops y el objetivo?

Stop Loss (SL): a 1.5 × ATR desde el precio de entrada.

Take Profit (TP): a 2.5 × ATR.

Trailing: el stop se va moviendo con la línea CE para proteger ganancias si el precio avanza a favor.

Slippage realista en stops: añade 12 bps de “empeoramiento” cuando salta un stop (simula ejecución real).

Cooldown: cuando cierra una operación, espera 3 velas antes de volver a entrar (evita “metralleta”).

¿Cuánto compra/vende (tamaño de posición)?

Calcula la distancia al stop (SL) usando el ATR.

Arriesga aprox. 2% del capital (PERCENT_RISK_ATR = 0.02):
tamaño ≈ (cash * 2%) / distancia_stop

Aplica límites realistas:

Apalancamiento máx.: 3×.

Límite de liquidez: no superar 1% del $volumen EMA de la vela.

Mínimo de orden del exchange: 11 USDT (si la orden es más pequeña, no opera).

Redondeo a 4 decimales (tamaños tipo 0.0001).

Comisiones: aplica 0.05% (taker) en la entrada y en la salida.

Resultado: si el mercado está tranquilo o tu cuenta es pequeña, el bot puede decidir no entrar porque la orden quedaría por debajo de 11 USDT o por límites de liquidez.

¿Cómo decide la IA (en simple)?

Agrupa patrones de 64 velas en 20 grupos (clusters).

Cada grupo tiene una probabilidad suavizada de ganar (usa una Beta(8,8) para no sobreconfiar).

Cuando hay señal técnica, revisa a qué grupo se parece más el patrón actual:

Si la probabilidad ≥ umbral (≈ 40.5% con tus TP/SL), acepta la operación.

Si no, la filtra y no entra.

(En vivo o en walk-forward puede “re-entrenar” usando solo datos recientes para adaptarse.)

Resumen del flujo del bot

Espera la señal CE + ADX (y EMA si la usas).

Si hay señal, la IA revisa si el patrón es lo bastante bueno.

Calcula tamaño con riesgo ≈ 2% y límites reales.

Entra (si supera mínimo de 11 USDT), aplica comisión.

Gestiona la salida con SL/TP y trailing CE (+ slippage en stops).

Cierra, anota PnL y hace cooldown de 3 velas.

Repite.
