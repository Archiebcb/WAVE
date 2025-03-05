from flask import Flask, render_template, request
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
    avg_gain = np.mean(gains[:period]) if np.any(gains[:period]) else 0
    avg_loss = np.mean(losses[:period]) if np.any(losses[:period]) else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs)) if avg_loss != 0 else 0
    rsi_values = [rsi]
    for i in range(period, len(prices) - 1):
        gain = gains[i - 1]
        loss = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs)) if avg_loss != 0 else 0
        rsi_values.append(rsi)
    return rsi_values

def calculate_macd(prices, slow=26, fast=12, signal=9):
    exp1 = np.convolve(prices, np.ones(fast) / fast, mode='valid')
    exp2 = np.convolve(prices, np.ones(slow) / slow, mode='valid')
    macd = exp1[-len(exp2):] - exp2
    signal_line = np.convolve(macd, np.ones(signal) / signal, mode='valid')
    return macd[-len(signal_line):].tolist(), signal_line.tolist()

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    sma = np.convolve(prices, np.ones(period) / period, mode='valid')
    rolling_std = np.std(prices[:period])
    upper_band = [sma[0] + (std_dev * rolling_std)]
    lower_band = [sma[0] - (std_dev * rolling_std)]
    for i in range(period, len(prices)):
        window = prices[i-period+1:i+1]
        rolling_std = np.std(window)
        upper_band.append(sma[i-period+1] + (std_dev * rolling_std))
        lower_band.append(sma[i-period+1] - (std_dev * rolling_std))
    return sma.tolist(), upper_band, lower_band

def calculate_sma_ema(prices, sma_period=20, ema_period=12):
    sma = np.convolve(prices, np.ones(sma_period) / sma_period, mode='valid').tolist()
    ema = [prices[0]]
    multiplier = 2 / (ema_period + 1)
    for i in range(1, len(prices)):
        ema.append(prices[i] * multiplier + ema[-1] * (1 - multiplier))
    return sma, ema

def calculate_stochastic(prices, k_period=14, d_period=3):
    low_min = np.minimum.accumulate(prices)
    high_max = np.maximum.accumulate(prices)
    k = []
    for i in range(len(prices) - k_period + 1):
        window_low = low_min[i + k_period - 1]
        window_high = high_max[i + k_period - 1]
        if window_high > window_low:
            k.append(100 * (prices[i + k_period - 1] - window_low) / (window_high - window_low))
        else:
            k.append(50.0)
    d = np.convolve(k, np.ones(d_period) / d_period, mode='valid').tolist()
    return k, d

def calculate_obv(prices, volumes):
    obv = [0]
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            obv.append(obv[-1] + volumes[i])
        elif prices[i] < prices[i-1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    return obv

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
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=30"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        prices = [price[1] for price in data['prices']]
        volumes = [volume[1] for volume in data['total_volumes']]
        labels = [f"Day {i+1}" for i in range(len(prices))]
        rsi_values = calculate_rsi(prices)
        macd_line, signal_line = calculate_macd(prices)
        sma, ema = calculate_sma_ema(prices)
        upper_bb, lower_bb = calculate_bollinger_bands(prices)[1:]
        k_stoch, d_stoch = calculate_stochastic(prices)
        obv = calculate_obv(prices, volumes)
        elliott_analysis = analyze_elliott_waves(prices)
        return labels, prices, volumes, rsi_values, macd_line, signal_line, sma, ema, upper_bb, lower_bb, k_stoch, d_stoch, obv, elliott_analysis
    print(f"Error fetching data for {coin_id}: {response.status_code} - {response.text}")
    return [], [], [], [], [], [], [], [], [], [], [], [], [], "Error fetching historical data."

def get_real_time_price(coin_id):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get(coin_id, {}).get('usd', 0)
    return 0

def send_email(subject, body, to_email):
    # Gmail SMTP settings (replace with your App Password)
    gmail_user = 'your-email@gmail.com'
    gmail_password = 'your-app-password'  # Use App Password, not regular password

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
                           user_email or "your-email@gmail.com")  # Default to your email for testing
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
                 chart_obv, elliott_analysis) = get_historical_data(selected_coin_id)
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
                          chart_obv=chart_obv, elliott_analysis=elliott_analysis,
                          real_time_price=real_time_price, portfolio_value=portfolio_value,
                          alert_threshold=alert_threshold)

if __name__ == "__main__":
    app.run(debug=True)