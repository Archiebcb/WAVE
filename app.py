from flask import Flask, render_template, request
import requests
import numpy as np
import smtplib
from email.mime.text import MIMEText
import time
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Email configuration (replace with your details)
EMAIL_ADDRESS = "your-email@gmail.com"  # Replace with your Gmail
EMAIL_PASSWORD = "your-app-password"    # Replace with your App Password
EMAIL_SERVER = "smtp.gmail.com"
EMAIL_PORT = 587

# Simple in-memory portfolio (for demo; use a DB in production)
PORTFOLIO = {}

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

def get_in_depth_analysis(prices, volumes, rsi_values, macd_line, signal_line, k_stoch, d_stoch):
    analysis = "In-Depth Market Analysis:\n\n"
    
    # Market Trend Overview
    price_change = (prices[-1] - prices[0]) / prices[0] * 100
    trend = "Uptrend" if price_change > 5 else "Downtrend" if price_change < -5 else "Sideways"
    analysis += f"1. Market Trend: {trend} ({price_change:.2f}% change over 30 days)\n"

    # Volatility Assessment
    volatility = np.std(np.diff(prices)) / np.mean(prices) * 100
    analysis += f"2. Volatility: {volatility:.2f}% (Average price fluctuation)\n"

    # Support and Resistance Levels
    if len(prices) > 10:
        support = min(prices[-10:])
        resistance = max(prices[-10:])
        analysis += f"3. Support/Resistance: Support at ${support:.2f}, Resistance at ${resistance:.2f} (last 10 days)\n"
    else:
        analysis += "3. Support/Resistance: Insufficient data for levels.\n"

    # Momentum Summary
    rsi_avg = np.mean(rsi_values[-5:]) if rsi_values else 50
    macd_diff = macd_line[-1] - signal_line[-1] if macd_line and signal_line else 0
    stoch_k = k_stoch[-1] if k_stoch else 50
    momentum = "Strong" if (rsi_avg > 70 or macd_diff > 0 or stoch_k > 80) else "Weak" if (rsi_avg < 30 or macd_diff < 0 or stoch_k < 20) else "Neutral"
    analysis += f"4. Momentum: {momentum} (RSI: {rsi_avg:.2f}, MACD Diff: {macd_diff:.2f}, Stochastic %K: {stoch_k:.2f})\n"

    # Volume Analysis
    volume_trend = "Increasing" if volumes[-1] > np.mean(volumes[-5:]) else "Decreasing"
    analysis += f"5. Volume Trend: {volume_trend} (Current: {volumes[-1]:,.0f}, 5-day Avg: {np.mean(volumes[-5:]):,.0f})\n"

    return analysis

def get_historical_data(coin_id):
    logger.debug(f"Fetching historical data for coin_id: {coin_id}")
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=30"
    time.sleep(1)  # Add delay to respect rate limit
    response = requests.get(url)
    logger.debug(f"API response status: {response.status_code}, text: {response.text}")
    if response.status_code == 200:
        data = response.json()
        prices = [price[1] for price in data['prices']]
        volumes = [volume[1] for volume in data['total_volumes']]
        labels = [f"Day {i+1}" for i in range(len(prices))]
        rsi_values = calculate_rsi(prices)
        macd_line, signal_line = calculate_macd(prices)
        sma, ema = calculate_sma_ema(prices)
        upper_bb, lower_bb = calculate_bollinger_bands(prices)[1:]  # Skip SMA for now
        k_stoch, d_stoch = calculate_stochastic(prices)
        obv = calculate_obv(prices, volumes)
        elliott_analysis = analyze_elliott_waves(prices)
        in_depth_analysis = get_in_depth_analysis(prices, volumes, rsi_values, macd_line, signal_line, k_stoch, d_stoch)
        return labels, prices, volumes, rsi_values, macd_line, signal_line, sma, ema, upper_bb, lower_bb, k_stoch, d_stoch, obv, elliott_analysis, in_depth_analysis
    logger.error(f"Failed to fetch data for {coin_id}: {response.status_code} - {response.text}")
    return [], [], [], [], [], [], [], [], [], [], [], [], [], "Error fetching historical data.", "Error fetching historical data."

def get_real_time_price(coin_id):
    logger.debug(f"Fetching real-time price for coin_id: {coin_id}")
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    max_retries = 5
    retry_delay = 1  # Initial delay in seconds
    for attempt in range(max_retries):
        time.sleep(retry_delay)  # Respect rate limit
        response = requests.get(url)
        logger.debug(f"Real-time price API attempt {attempt + 1}, status: {response.status_code}, text: {response.text}")
        if response.status_code == 200:
            data = response.json()
            price = data.get(coin_id, {}).get('usd', 0)
            return price
        elif response.status_code == 429:
            logger.warning(f"Rate limit hit, retrying in {retry_delay} seconds...")
            retry_delay *= 2  # Exponential backoff
            if retry_delay > 10:
                retry_delay = 10  # Cap at 10 seconds
            continue
        else:
            logger.error(f"Unexpected status code {response.status_code} for {coin_id}")
            break
    logger.error(f"Failed to fetch real-time price for {coin_id} after {max_retries} attempts")
    return 0

def send_email(subject, body, to_email):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email

    try:
        with smtplib.SMTP(EMAIL_SERVER, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info(f"Email sent to {to_email}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

# Store emails locally (for demo; use a DB in production)
EMAIL_LIST = []

@app.route("/", methods=["GET", "POST"])
def home():
    global EMAIL_LIST, PORTFOLIO
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
    in_depth_analysis = ""
    real_time_price = 0

    # Handle form submission (POST)
    if request.method == "POST":
        selected_coin_id = request.form.get("ticker")
        logger.debug(f"Selected coin_id from form: {selected_coin_id}")
        portfolio_email = request.form.get("portfolio_email")
        alert_email = request.form.get("alert_email")
        alert_threshold = request.form.get("alert_threshold")

        if selected_coin_id:
            url = f"https://api.coingecko.com/api/v3/coins/{selected_coin_id}"
            response = requests.get(url)
            logger.debug(f"Coin info response status: {response.status_code}")
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
                 chart_obv, elliott_analysis, in_depth_analysis) = get_historical_data(selected_coin_id)
                real_time_price = get_real_time_price(selected_coin_id)

                # Portfolio email (daily summary) - Test with frequent check
                if portfolio_email and datetime.now().minute % 1 == 0:  # Check every minute for testing
                    portfolio_value = sum(qty * get_real_time_price(coin_id) for coin_id, qty in PORTFOLIO.items() if get_real_time_price(coin_id))
                    send_email("Daily Portfolio Update", f"Portfolio Value for {portfolio_email}: ${portfolio_value:.2f}", portfolio_email)

                # Alert email
                if alert_email and alert_threshold and real_time_price >= float(alert_threshold):
                    send_email(f"Price Alert for {crypto_data['ticker']}", f"Price reached ${real_time_price} (Threshold: ${alert_threshold})", alert_email)
            else:
                crypto_data = {"error": "Unable to fetch data for this coin"}

    # Render the template (GET or after POST processing)
    return render_template("index.html", crypto_data=crypto_data, coins=COIN_LIST,
                          chart_labels=chart_labels, chart_prices=chart_prices,
                          chart_volumes=chart_volumes, chart_rsi=chart_rsi,
                          chart_macd=chart_macd, chart_signal=chart_signal,
                          chart_sma=chart_sma, chart_ema=chart_ema,
                          chart_upper_bb=chart_upper_bb, chart_lower_bb=chart_lower_bb,
                          chart_k_stoch=chart_k_stoch, chart_d_stoch=chart_d_stoch,
                          chart_obv=chart_obv, elliott_analysis=elliott_analysis,
                          in_depth_analysis=in_depth_analysis, real_time_price=real_time_price)

@app.route("/portfolio", methods=["GET", "POST"])
def portfolio():
    global PORTFOLIO
    if request.method == "POST":
        coin_id = request.form.get("coin_id")
        quantity = request.form.get("quantity")
        if coin_id and quantity:
            PORTFOLIO[coin_id] = float(quantity)
        elif request.form.get("remove_coin"):
            coin_id = request.form.get("remove_coin")
            PORTFOLIO.pop(coin_id, None)
    return render_template("portfolio.html", portfolio=PORTFOLIO, coins=COIN_LIST)

@app.route("/backtesting")
def backtesting():
    return render_template("backtesting.html")

@app.route("/news")
def news():
    # Simple mock news (replace with RSS feed or X scraping later)
    news_items = ["Bitcoin surges 5%!", "Ethereum upgrade delayed", "XRP wins legal battle"]
    return render_template("news.html", news_items=news_items)

@app.route("/learning")
def learning():
    return render_template("learning.html")

if __name__ == "__main__":
    app.run(debug=True)