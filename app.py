from flask import Flask, render_template, request
import requests

app = Flask(__name__)

# Fetch CoinGecko's list of verified coins once at startup
def get_coingecko_coins():
    url = "https://api.coingecko.com/api/v3/coins/list"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # List of dicts with 'id', 'symbol', 'name'
    return []

# Store the coin list globally (fetched once when the app starts)
COIN_LIST = get_coingecko_coins()

@app.route("/", methods=["GET", "POST"])
def home():
    crypto_data = None
    if request.method == "POST":
        # Get the CoinGecko ID from the dropdown
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
        else:
            crypto_data = {"error": "Unable to fetch data for this coin"}

    # Pass the coin list to the template for the dropdown
    return render_template("index.html", crypto_data=crypto_data, coins=COIN_LIST)

if __name__ == "__main__":
    app.run(debug=True)