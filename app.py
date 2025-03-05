from flask import Flask, render_template, request, redirect
import requests
import numpy as np
import smtplib
from email.mime.text import MIMEText
import json
import os
from datetime import datetime

app = Flask(__name__)

def get_coingecko_coins():
    url = "https://api.coingecko.com/api/v3/coins/list"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return []

COIN_LIST = get_coingecko_coins()

def calculate_rsi(prices, period=14):
    changes = np.diff(prices)
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)
    avg_gain = np.mean(gains[:period]) if len(prices) > period and np.any(gains[:period]) else 0
    avg_loss = np.mean(losses[:period]) if len(prices) > period and np.any(losses[:period]) else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs)) if avg_loss != 0 else 0
    rsi_values = [rsi]
    for i in range(period, len(prices) - 1):
        gain = gains[i - 1] if i - 1 < len(gains) else 0
        loss = losses[i - 1] if i - 1 < len(losses) else 0
        avg_gain = (avg_gain * (period - 1) + gain) / period if avg_gain else 0
        avg_loss = (avg_loss * (period - 1) + loss) / period if avg_loss else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs)) if avg_loss != 0 else 0
        rsi_values.append(rsi)
    return list(np.pad(rsi_values, (0, len(prices) - len(rsi_values)), mode='constant'))

def calculate_macd(prices, slow=26, fast=12, signal=9):
    exp1 = np.convolve(prices, np.ones(fast) / fast, mode='valid')
    exp2 = np.convolve(prices, np.ones(slow) / slow, mode='valid')
    macd = exp1[-len(exp2):] - exp2 if exp2.size else np.zeros(len(prices) - fast + 1)
    signal_line = np.convolve(macd, np.ones(signal) / signal, mode='valid') if macd.size >= signal else np.zeros(len(macd) - signal + 1)
    return list(np.pad(macd, (0, len(prices) - len(macd)), mode='constant')), list(np.pad(signal_line, (0, len(prices) - len(signal_line)), mode='constant'))

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    if len(prices) < period:
        return [0] * len(prices), [0] * len(prices), [0] * len(prices)
    sma = np.convolve(prices, np.ones(period) / period, mode='valid')
    rolling_std = np.std(prices[:period]) if len(prices[:period]) == period else 0
    upper_band = [sma[0] + (std_dev * rolling_std) if rolling_std and i < len(sma) else 0 for i in range(len(prices))]
    lower_band = [sma[0] - (std_dev * rolling_std) if rolling_std and i < len(sma) else 0 for i in range(len(prices))]
    return list(np.pad(sma, (0, len(prices) - len(sma)), mode='constant')), upper_band, lower_band

def calculate_sma_ema(prices, sma_period=20, ema_period=12):
    if len(prices) < sma_period:
        return [0] * len(prices), [0] * len(prices)
    sma = np.convolve(prices, np.ones(sma_period) / sma_period, mode='valid').tolist()
    ema = [prices[0]] if prices else [0]
    multiplier = 2 / (ema_period + 1)
    for i in range(1, len(prices)):
        ema.append(prices[i] * multiplier + ema[-1] * (1 - multiplier) if i < ema_period else ema[-1])
    return sma + [0] * (len(prices) - len(sma)), ema

def calculate_stochastic(prices, k_period=14, d_period=3):
    if len(prices) < k_period:
        return [50.0] * len(prices), [50.0] * len(prices)
    low_min = np.minimum.accumulate(prices)
    high_max = np.maximum.accumulate(prices)
    k = []
    for i in range(len(prices) - k_period + 1):
        window_low = low_min[i + k_period - 1] if i + k_period - 1 < len(low_min) else 0
        window_high = high_max[i + k_period - 1] if i + k_period - 1 < len(high_max) else 0
        if window_high > window_low:
            k.append(100 * (prices[i + k_period - 1] - window_low) / (window_high - window_low) if i + k_period - 1 < len(prices) else 50.0)
        else:
            k.append(50.0)
    d = np.convolve(k, np.ones(d_period) / d_period, mode='valid').tolist()
    return [50.0] * (k_period - 1) + k + [50.0] * (len(prices) - len(k) - k_period + 1), [50.0] * (k_period + d_period - 2) + d + [50.0] * (len(prices) - len(d) - k_period - d_period + 2)

def calculate_obv(prices, volumes):
    obv = [0]
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            obv.append(obv[-1] + volumes[i] if i < len(volumes) else obv[-1])
        elif prices[i] < prices[i-1]:
            obv.append(obv[-1] - volumes[i] if i < len(volumes) else obv[-1])
        else:
            obv.append(obv[-1])
    return obv + [0] * (len(prices) - len(obv))

def calculate_ichimoku(prices, high_prices, low_prices):
    if len(prices) < 52:
        return [0] * len(prices), [0] * len(prices), [0] * len(prices), [0] * len(prices)
    
    tenkan_sen = [0] * 9
    kijun_sen = [0] * 26
    senkou_span_a = [0] * 26
    senkou_span_b = [0] * 52

    for i in range(9, len(prices)):
        tenkan_window = high_prices[i-9:i+1] + low_prices[i-9:i+1] if i + 1 <= len(high_prices) else []
        if tenkan_window and len(tenkan_window) >= 10:
            tenkan_sen.append((max(tenkan_window) + min(tenkan_window)) / 2)
        else:
            tenkan_sen.append(0)

    for i in range(26, len(prices)):
        kijun_window = high_prices[i-26:i+1] + low_prices[i-26:i+1] if i + 1 <= len(high_prices) else []
        if kijun_window and len(kijun_window) >= 27:
            kijun_sen.append((max(kijun_window) + min(kijun_window)) / 2)
        else:
            kijun_sen.append(0)

    for i in range(26, len(prices)):
        if len(tenkan_sen) > i - 26 and len(kijun_sen) > i - 26:
            senkou_a = (tenkan_sen[i - 26] + kijun_sen[i - 26]) / 2
            senkou_span_a.append(senkou_a)
        else:
            senkou_span_a.append(0)

    for i in range(52, len(prices)):
        senkou_b_window = high_prices[i-52:i+1] + low_prices[i-52:i+1] if i + 1 <= len(high_prices) else []
        if senkou_b_window and len(senkou_b_window) >= 53:
            senkou_b = (max(senkou_b_window) + min(senkou_b_window)) / 2
            senkou_span_b.append(senkou_b)
        else:
            senkou_span_b.append(0)

    # Pad to match prices length
    tenkan_sen = [0] * (len(prices) - len(tenkan_sen)) + tenkan_sen
    kijun_sen = [0] * (len(prices) - len(kijun_sen)) + kijun_sen
    senkou_span_a = [0] * (len(prices) - len(senkou_span_a)) + senkou_span_a
    senkou_span_b = [0] * (len(prices) - len(senkou_span_b)) + senkou_span_b

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b

def calculate_atr(high_prices, low_prices, close_prices, period=14):
    tr = []
    for i in range(1, len(high_prices)):
        tr.append(max(high_prices[i] - low_prices[i], 
                      abs(high_prices[i] - close_prices[i-1]), 
                      abs(low_prices[i] - close_prices[i-1])))
    atr = [np.mean(tr[:period])] if tr and len(tr) >= period else [0]
    for i in range(period, len(tr)):
        atr.append((atr[-1] * (period - 1) + tr[i]) / period)
    return [0] * (len(high_prices) - len(atr)) + atr

def calculate_rvi(prices, period=14):
    price_changes = np.diff(prices)
    up_moves = np.where(price_changes > 0, price_changes, 0)
    down_moves = np.where(price_changes < 0, -price_changes, 0)
    rvi_num = np.convolve(up_moves, np.ones(period) / period, mode='valid')
    rvi_den = np.convolve(down_moves, np.ones(period) / period, mode='valid')
    rvi = [r * 100 / (r + 1) if r_den != 0 else 50 for r, r_den in zip(rvi_num, rvi_den) if r_den != 0]
    return list(np.pad([50] * (period - 1) + rvi + [50] * (len(prices) - len(rvi) - period + 1), (0, 0), mode='constant'))

def calculate_ad_line(high_prices, low_prices, close_prices, volumes, prices):
    adl = [0]
    for i in range(1, len(close_prices)):
        if high_prices[i] != low_prices[i]:  # Avoid division by zero
            money_flow_multiplier = ((close_prices[i] - low_prices[i]) - (high_prices[i] - close_prices[i])) / (high_prices[i] - low_prices[i])
            money_flow_volume = money_flow_multiplier * volumes[i] if i < len(volumes) else 0
            adl.append(adl[-1] + money_flow_volume)
        else:
            adl.append(adl[-1])
    return adl + [0] * (len(prices) - len(adl))

def calculate_parabolic_sar(high_prices, low_prices, prices, start=0.02, increment=0.02, maximum=0.2):
    sar = [low_prices[0] * (1 - start)] if low_prices and len(low_prices) > 0 else [0]
    ep = high_prices[0] if high_prices and len(high_prices) > 0 else 0
    af = start
    trend = 1
    for i in range(1, len(high_prices)):
        if trend == 1:
            sar.append(min(sar[-1] + af * (ep - sar[-1]), low_prices[i] if i < len(low_prices) else sar[-1]))
            if high_prices[i] > ep if i < len(high_prices) else False:
                ep = high_prices[i]
                af = min(af + increment, maximum)
            if low_prices[i] < sar[-1] if i < len(low_prices) else False:
                trend = -1
                sar[-1] = ep
                ep = low_prices[i] if i < len(low_prices) else ep
                af = start
        else:
            sar.append(max(sar[-1] + af * (ep - sar[-1]), high_prices[i] if i < len(high_prices) else sar[-1]))
            if low_prices[i] < ep if i < len(low_prices) else False:
                ep = low_prices[i]
                af = min(af + increment, maximum)
            if high_prices[i] > sar[-1] if i < len(high_prices) else False:
                trend = 1
                sar[-1] = ep
                ep = high_prices[i]
                af = start
    return sar + [0] * (len(prices) - len(sar))

def analyze_elliott_waves(prices):
    if len(prices) < 8:
        return "Insufficient data for Elliott Wave analysis."
    
    peaks = []
    troughs = []
    for i in range(1, len(prices) - 1):
        if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
            peaks.append((i, prices[i]))
        elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
            troughs.append((i, prices[i]))
    
    if not peaks or not troughs:
        return "No clear wave pattern detected."

    wave_analysis = "Elliott Wave Breakdown:\n"
    if len(peaks) >= 5 and len(troughs) >= 3:
        wave_analysis += "Potential 5-Wave Impulse:\n"
        for i in range(min(5, len(peaks))):
            wave_analysis += f"  Wave {i+1}: Peak at Day {peaks[i][0]+1} - ${peaks[i][1]:.2f}\n"
        wave_analysis += "Potential 3-Wave Correction:\n"
        for i in range(min(3, len(troughs))):
            wave_analysis += f"  Wave A/B/C {i+1}: Trough at Day {troughs[i][0]+1} - ${troughs[i][1]:.2f}\n"
    else:
        wave_analysis += "Incomplete wave pattern. Requires more data or clearer trends."

    return wave_analysis

def get_historical_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=60"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        prices = [price[1] for price in data['prices']]
        volumes = [volume[1] for volume in data['total_volumes']]
        ohlc = data.get('ohlc', None)  # Check if OHLC is available
        if ohlc and len(ohlc) == len(prices):
            high_prices = [h[2] for h in ohlc]
            low_prices = [l[3] for l in ohlc]
            close_prices = [c[4] for c in ohlc]
        else:
            high_prices = prices.copy()  # Fallback to prices if no OHLC
            low_prices = prices.copy()
            close_prices = prices.copy()
        labels = [f"Day {i+1}" for i in range(len(prices))]
        rsi_values = calculate_rsi(prices)
        macd_line, signal_line = calculate_macd(prices)
        sma, ema = calculate_sma_ema(prices)
        upper_bb, lower_bb = calculate_bollinger_bands(prices)[1:]
        k_stoch, d_stoch = calculate_stochastic(prices)
        obv = calculate_obv(prices, volumes)
        tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(prices, high_prices, low_prices)
        atr = calculate_atr(high_prices, low_prices, close_prices)
        rvi = calculate_rvi(prices)
        ad_line = calculate_ad_line(high_prices, low_prices, close_prices, volumes, prices)
        psar = calculate_parabolic_sar(high_prices, low_prices, prices)
        elliott_analysis = analyze_elliott_waves(prices)
        # Ensure all datasets match the length of labels
        max_len = len(labels)
        return (labels, 
                prices, 
                volumes, 
                rsi_values, 
                macd_line, 
                signal_line, 
                sma, 
                ema, 
                upper_bb, 
                lower_bb, 
                k_stoch, 
                d_stoch, 
                obv, 
                tenkan, 
                kijun, 
                senkou_a, 
                senkou_b, 
                atr, 
                rvi, 
                ad_line, 
                psar, 
                elliott_analysis)
    print(f"Error fetching data for {coin_id}: {response.status_code} - {response.text}")
    return ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], "Error fetching historical data.")

def get_real_time_price(coin_id):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get(coin_id, {}).get('usd', 0)
    return 0

def send_email(subject, body, to_email):
    gmail_user = 'your-email@gmail.com'  # Replace with your Gmail
    gmail_password = 'your-app-password'  # Replace with your App Password

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = gmail_user
    msg['To'] = to_email

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(gmail_user, gmail_password)
        server.sendmail(gmail_user, to_email, msg.as_string())
        server.quit()
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def save_email(email):
    email_file = 'emails.txt'
    with open(email_file, 'a') as f:
        f.write(f"{email}\n")
    print(f"Email {email} saved to {email_file}")

@app.route("/", methods=["GET", "POST"])
def home():
    crypto_data = None
    chart_labels = []
    chart_prices = []
    chart_volumes = []
    chart_rsi = []
    chart_macd = []
    chart_signal = []
    chart_sma = []
    chart_ema = []
    chart_upper_bb = []
    chart_lower_bb = []
    chart_k_stoch = []
    chart_d_stoch = []
    chart_obv = []
    chart_tenkan = []
    chart_kijun = []
    chart_senkou_a = []
    chart_senkou_b = []
    chart_atr = []
    chart_rvi = []
    chart_ad_line = []
    chart_psar = []
    elliott_analysis = ""
    real_time_price = 0
    portfolio_value = 0
    alert_threshold = None
    selected_coin_id = request.form.get("ticker") if request.method == "POST" else None
    portfolio = json.loads(request.cookies.get('portfolio', '[]')) if request.cookies.get('portfolio') else []

    if request.method == "POST":
        action = request.form.get("action")
        if action == "add_portfolio":
            ticker = request.form.get("portfolioTicker").lower()
            amount = float(request.form.get("portfolioAmount"))
            portfolio.append({"ticker": ticker, "amount": amount})
            response = app.make_response(redirect(request.url))
            response.set_cookie('portfolio', json.dumps(portfolio))
            return response
        elif action == "set_alert":
            alert_threshold = float(request.form.get("alertPrice"))
            user_email = request.form.get("userEmail")
            if user_email:
                save_email(user_email)
            if real_time_price and alert_threshold and real_time_price >= alert_threshold:
                send_email(f"Price Alert for {selected_coin_id.upper()}",
                           f"The price of {selected_coin_id.upper()} has reached ${real_time_price} at {datetime.now()}.",
                           user_email or "your-email@gmail.com")
        elif selected_coin_id:
            url = f"https://api.coingecko.com/api/v3/coins/{selected_coin_id}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                crypto_data = {
                    "ticker": data["symbol"].upper(),
                    "name": data["name"],
                    "price": f"${data['market_data']['current_price']['usd']:.2f}",
                    "market_cap": f"${data['market_data']['market_cap']['usd']:,.0f}",
                    "volume_24h": f"${data['market_data']['total_volume']['usd']:,.0f}"
                }
                (chart_labels, chart_prices, chart_volumes, chart_rsi, chart_macd, chart_signal,
                 chart_sma, chart_ema, chart_upper_bb, chart_lower_bb, chart_k_stoch, chart_d_stoch,
                 chart_obv, chart_tenkan, chart_kijun, chart_senkou_a, chart_senkou_b,
                 chart_atr, chart_rvi, chart_ad_line, chart_psar, elliott_analysis) = get_historical_data(selected_coin_id)
                real_time_price = get_real_time_price(selected_coin_id)
                if portfolio:
                    portfolio_value = sum(item["amount"] * get_real_time_price(item["ticker"]) for item in portfolio if get_real_time_price(item["ticker"]))
            else:
                crypto_data = {"error": "Unable to fetch data for this coin"}

    return render_template("index.html", crypto_data=crypto_data, coins=COIN_LIST,
                          chart_labels=chart_labels, chart_prices=chart_prices,
                          chart_volumes=chart_volumes, chart_rsi=chart_rsi,
                          chart_macd=chart_macd, chart_signal=chart_signal,
                          chart_sma=chart_sma, chart_ema=chart_ema,
                          chart_upper_bb=chart_upper_bb, chart_lower_bb=chart_lower_bb,
                          chart_k_stoch=chart_k_stoch, chart_d_stoch=chart_d_stoch,
                          chart_obv=chart_obv, chart_tenkan=chart_tenkan,
                          chart_kijun=chart_kijun, chart_senkou_a=chart_senkou_a,
                          chart_senkou_b=chart_senkou_b, chart_atr=chart_atr,
                          chart_rvi=chart_rvi, chart_ad_line=chart_ad_line,
                          chart_psar=chart_psar, elliott_analysis=elliott_analysis,
                          real_time_price=real_time_price, portfolio_value=portfolio_value,
                          alert_threshold=alert_threshold)

if __name__ == "__main__":
    app.run(debug=True)