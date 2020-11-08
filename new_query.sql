WITH avg_hum AS (
    SELECT UTCDateTime, AVG(ActualValue) AS humidity
    FROM CEVAC_WATT_HUM_HIST
    GROUP BY UTCDateTime
),
avg_temp AS (
    SELECT UTCDateTime, AVG(ActualValue) AS temperature
    FROM CEVAC_WATT_TEMP_HIST
    GROUP BY UTCDateTime
),
avg_co2 AS (
    SELECT UTCDateTime, AVG(ActualValue) AS co2
    FROM CEVAC_WATT_CO2_HIST
    GROUP BY UTCDateTime
)
SELECT DISTINCT DATEPART(WEEKDAY, ps.UTCDateTime) AS 'weekday', DATEPART(HOUR, ps.UTCDateTime) AS 'hour', wfs.SUM_total AS 'occupancy', avg_hum.humidity, avg_temp.temperature, avg_co2.co2, ps.Total_Usage AS 'power'
FROM CEVAC_WATT_POWER_SUMS_HIST AS ps
INNER JOIN CEVAC_WATT_WAP_FLOOR_SUMS_HIST AS wfs ON wfs.UTCDateTime = ps.UTCDateTime
INNER JOIN avg_hum ON avg_hum.UTCDateTime = ps.UTCDateTime
INNER JOIN avg_temp ON avg_temp.UTCDateTime = ps.UTCDateTime
INNER JOIN avg_co2 ON avg_co2.UTCDateTime = ps.UTCDateTime
