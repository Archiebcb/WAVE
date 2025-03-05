from flask import Flask, render_template, request
import requests

app = Flask(__name__)

def get_coingecko_coins():
    url = "https://api.coingecko.com/api/v3/coins/list"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return []

COIN_LIST = get_coingecko_coins()

def get_historical_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=7"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        prices = data['prices']
        labels = [f"Day {i+1}" for i in range(len(prices))]
        values = [price[1] for price in prices]
        return labels, values
    print(f"Error fetching data for {coin_id}: {response.status_code} - {response.text}")
    return [], []

@app.route("/", methods=["GET", "POST"])
def home():
    crypto_data = None
    chart_labels = []
    chart_data = []
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
            chart_labels, chart_data = get_historical_data(coin_id)
        else:
            crypto_data = {"error": "Unable to fetch data for this coin"}

    return render_template("index.html", crypto_data=crypto_data, coins=COIN_LIST, chart_labels=chart_labels, chart_data=chart_data)

if __name__ == "__main__":
    app.run(debug=True)