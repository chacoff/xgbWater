WITH FilteredMeasures AS (
    SELECT Timestamp, Filename
    FROM Measures
    WHERE Timestamp < '2024-12-20 15:22:00,000' AND Filename = 'Pass 3'
)

SELECT MIN(TimeStamp) AS TimeStamp, Filename
FROM FilteredMeasures
UNION ALL
SELECT MAX(TimeStamp) AS TimeStamp, Filename
FROM FilteredMeasures;