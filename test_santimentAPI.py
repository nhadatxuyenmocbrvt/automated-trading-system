import requests

# Public API key hoặc token của bạn – chính xác với GraphiQL_Advanced
API_KEY = "nukmw447n726adit_m3shnmr3m5o5fgge"  # ví dụ từ ảnh của bạn

def fetch_social_volume():
    url = "https://api.santiment.net/graphql"  # Endpoint đúng, dù bạn test từ graphiql_advanced

    headers = {
        "Authorization": f"Apikey {API_KEY}",  # Nếu là public key
        "Content-Type": "application/json"
    }

    query = """
    {
      getMetric(metric: "social_volume_total") {
        timeseriesData(
          slug: "bitcoin"
          from: "2024-04-01T00:00:00Z"
          to: "2024-04-03T00:00:00Z"
          interval: "1d"
        ) {
          datetime
          value
        }
      }
    }
    """

    response = requests.post(url, headers=headers, json={"query": query})
    try:
        data = response.json()
        if "errors" in data:
            print("❌ Lỗi API:", data["errors"])
        else:
            print("✅ Dữ liệu nhận được:")
            for point in data["data"]["getMetric"]["timeseriesData"]:
                print(f"{point['datetime']} – {point['value']}")
    except Exception as e:
        print("❌ Lỗi khi phân tích JSON:", e)
        print("Phản hồi thô:", response.text)

if __name__ == "__main__":
    fetch_social_volume()
