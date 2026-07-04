import pymongo

CONN_STR = "mongodb://localhost:27017/"
DB_NAME = "flight-booking"

# Connect to MongoDB using connection string
client = pymongo.MongoClient(CONN_STR)
db = client[DB_NAME]

# Return type should be types.Airport
def get_airport_code_by_city(city: str):
    # connect to collection
    airports_collection = db["airport-info"]
    print(f"Fetching airport code for city: {city}")
    airport = airports_collection.find_one({"city": city})
    if airport:
        print(f"Found airport: {airport}")
        airport.pop("_id")  # Remove MongoDB's ObjectId
        return airport
    return None

# Return flight fare information by departure_airport_code and arrival_airport_code
def get_flight_fare_info(departure_airport_code: str, arrival_airport_code: str):
    # connect to collection
    fare_collection = db["flight-fare"]
    print(f"Fetching flight fares from {departure_airport_code} to {arrival_airport_code}")
    flight = fare_collection.find({
        "departure_airport_code": departure_airport_code,
        "arrival_airport_code": arrival_airport_code
    })
    if flight:
        fares = []
        for fare in flight:
            fare.pop("_id")  # Remove MongoDB's ObjectId
            fares.append(fare)
        print(f"Found flight fares: {fares}")
        return fares
    return None

# Return flight details by flight_number
def get_flight_details(flight_number: str):
    # connect to collection
    flight_collection = db["flight-info"]
    flight_details = flight_collection.find_one({"flight_number": flight_number})
    if flight_details:
        flight_details.pop("_id")  # Remove MongoDB's ObjectId
        return flight_details
    return None
