import sys
import requests

def main():
    err_check()
    coins = float(sys.argv[1])
    total_btc = convert(coins)
    print(f"${total_btc}")
    return


def convert(coins):
    r = requests.get('https://api.coindesk.com/v1/bpi/currentprice.json')
    btc_json = r.json()
    btc = btc_json["bpi"]["USD"]["rate"].replace(",", "")
    total_btc = coins * float(btc)
    total_btc = '{:,}'.format(total_btc)
    return total_btc


def err_check():
    for s in sys.argv[1:]:
        try:
            s = float(s)
        except ValueError:
            sys.exit("Command-line arugment is not a number")
    if len(sys.argv) > 2:
        sys.exit("Invalid command-line argument")
    elif len(sys.argv) < 2:
        sys.exit("Missing command-line argument")
    elif len(sys.argv) == 2:
        return



main()