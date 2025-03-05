from flask import Flask, render_template, request
import requests
import numpy as np

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
    for i in range(period, len(prices) - 1):
        gain = gains[i - 1]
        loss = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs)) if avg_loss != 0 else 0
        yield rsi

def calculate_macd(prices, slow=26, fast=12, signal=9):
    exp1 = np.convolve(prices, np.ones(fast) / fast, mode='valid')
    exp2 = np.convolve(prices, np.ones(slow) / slow, mode='valid')
    macd = exp1[-len(exp2):] - exp2
    signal_line = np.convolve(macd, np.ones(signal) / signal, mode='valid')
    return macd[-len(signal_line):].tolist(), signal_line.tolist()

def analyze_elliott_waves(prices):
    if len(prices) < 8:  # Need at least 8 points for a basic 5-3 pattern
        return "Insufficient data for Elliott Wave analysis."
    
    # Simple peak/trough detection (basic approximation)
    peaks = []
    troughs = []
    for i in range(1, len(prices) - 1):
        if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
            peaks.append((i, prices[i]))
        elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
            troughs.append((i, prices[i]))
    
    if not peaks or not troughs:
        return "No clear wave pattern detected."

    # Approximate 5-wave impulse and 3-wave correction
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
        rsi_values = list(calculate_rsi(prices))
        macd_line, signal_line = calculate_macd(prices)
        elliott_analysis = analyze_elliott_waves(prices)
        return labels, prices, volumes, rsi_values, macd_line, signal_line, elliott_analysis
    print(f"Error fetching data for {coin_id}: {response.status_code} - {response.text}")
    return [], [], [], [], [], [], "Error fetching historical data."

@app.route("/", methods=["GET", "POST"])
def home():
    crypto_data = None
    chart_labels = []
    chart_prices = []
    chart_volumes = []
    chart_rsi = []
    chart_macd = []
    chart_signal = []
    elliott_analysis = ""
    if request.method == "POST":
        coin_id = request.form["ticker"]
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
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
            chart_labels, chart_prices, chart_volumes, chart_rsi, chart_macd, chart_signal, elliott_analysis = get_historical_data(coin_id)
        else:
            crypto_data = {"error": "Unable to fetch data for this coin"}

    return render_template("index.html", crypto_data=crypto_data, coins=COIN_LIST, 
                          chart_labels=chart_labels, chart_prices=chart_prices, 
                          chart_volumes=chart_volumes, chart_rsi=chart_rsi, 
                          chart_macd=chart_macd, chart_signal=chart_signal, 
                          elliott_analysis=elliott_analysis)

if __name__ == "__main__":
    app.run(debug=True)