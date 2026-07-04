from mcp.server.fastmcp import FastMCP
import db_ops as ops

mcp = FastMCP("ABC Travel Agency")

# Add pydantic classes to structure responses
@mcp.tool()
def fetch_airport_code(city: str):
    """Return airport code alongwith country details when queried by City"""
    result = ops.get_airport_code_by_city(city)
    if result is None:
        return "Airport not found"
    else:
        return result

@mcp.tool()
def fetch_flight_fares(departure_airport_code: str, arrival_airport_code: str):
    """Return list of flight fares when queried by depature_airport_code and arrival_airport_code"""
    result = ops.get_flight_fare_info(departure_airport_code, arrival_airport_code)
    if result is None:
        return "No flight fares found"
    else:
        return result
    
@mcp.tool()
def fetch_flight_details(flight_number: str):
    """Return flight details when queried by flight_number"""
    result = ops.get_flight_details(flight_number)
    if result is None:
        return "Flight details not found"
    else:
        return result

if __name__ == "__main__":
    # To run as streamable http
    mcp.run(transport="streamable-http")