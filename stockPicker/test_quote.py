from futu import OpenQuoteContext

HOST = "127.0.0.1"
PORT = 11111   # OpenD port

def main():
    quote_ctx = OpenQuoteContext(host=HOST, port=PORT)

    ret, data = quote_ctx.get_market_snapshot(["HK.00700"])
    if ret == 0:
        print(data)
    else:
        print("Error:", data)

    quote_ctx.close()

if __name__ == "__main__":
    main()
