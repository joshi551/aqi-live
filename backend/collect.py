# Downloads the latest hourly data for every live city and appends it to data/live/<city>.csv
# Run it every day (Task Scheduler on Windows, cron on Linux/Mac) so the history keeps growing.
from common import CITY_SETS
import data_source as ds

for city in CITY_SETS["live"]:
    d = ds.load_city("live", city)
    print(city, "rows:", len(d), "latest:", d.index[-1], "|", ds.STATUS[city]["note"])
