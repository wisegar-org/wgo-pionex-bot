# Pionex Bot CE+ADX + IA de Patrones + Autoentrenador (archivo único)

## Pasos
1) Crea `.env` junto a `pionex_bot.py`:
```
PIONEX_API_KEY=TU_KEY
PIONEX_API_SECRET=TU_SECRET
```
2) Instala deps:
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
3) Prueba en seco (no opera):
```
python pionex_bot.py --dryrun
```
4) En vivo:
```
python pionex_bot.py
```

### Config rápida (variables dentro del script o como env)
- Símbolo/TF: BTC_USDT_PERP, 15m
- Riesgo: 2%/trade, leverage 3x, min notional 11 USDT
- Estrategia: CE(22,2.0) + ADX(14)>35, SL=1.5*ATR, TP=2.5*ATR
- IA: ventana 64 (log-returns, RSI, ADX), K=20, Beta(8,8), min_count=100
- Autoentrenador: reentrena cada 30 min con 180 días y actualiza modelo.
