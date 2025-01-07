package main

import (
	"database/sql"
	"fmt"
	_ "github.com/mattn/go-sqlite3"
	"os"
	"strings"
	"time"

	"github.com/muesli/clusters"
	"github.com/muesli/kmeans"
)

type db struct {
	database *sql.DB
}

var dataBase db = db{}

func main() {
	dataBase = db{}
	openError := dataBase.openDatabase()
	if openError != nil {
		return
	}

	timestamps, queryError := dataBase.getData()
	if queryError != nil {
		fmt.Printf("Error querying database: %v\n", queryError)
		return
	}

	var unixTimes []float64
	var timestampMap = make(map[float64]string)

	for _, ts := range timestamps {
		t, err := parseFlexibleTimestamp(ts)
		if err != nil {
			fmt.Printf("Error parsing timestamp: %v\n", err)
			return
		}

		unixTime := float64(t.UnixMilli()) // maybe uni nano?
		unixTimes = append(unixTimes, unixTime)
		timestampMap[unixTime] = ts // dict unixTime as key and value timestamp in string
	}

	firstUnixTime := unixTimes[0]
	firstTimestamp := timestampMap[firstUnixTime]
	fmt.Printf("First timestamp (string): %s\nFirst timestamp (unix): %.0f\n", firstTimestamp, firstUnixTime)

	// Clustering, convert 1D data to clusters.Observations
	var observations clusters.Observations
	var k int = 3

	for _, value := range unixTimes {
		observations = append(observations, clusters.Coordinates{value})
	}

	// Perform k-means clustering with 3 clusters
	km := kmeans.New()
	clustersPasses, err := km.Partition(observations, k)
	if err != nil {
		fmt.Println("Error partitioning data: ", err)
	}

	// Output results
	for i, c := range clustersPasses {
		fmt.Printf("Cluster %d:\n", i+1)
		for _, obs := range c.Observations {
			coordinates := obs.(clusters.Coordinates)
			fmt.Printf("  %.2f - %s\n", coordinates[0], timestampMap[coordinates[0]])
		}
		fmt.Printf("Centered at: %.3f\n", c.Center[0])
	}
}

func parseFlexibleTimestamp(ts string) (time.Time, error) {
	ts = strings.Replace(ts, ",", ".", 1) // Replace , with . due to Go parsing

	layouts := []string{
		"2006-01-02 15:04:05",     // no decimals
		"2006-01-02 15:04:05.0",   // 1 decimal
		"2006-01-02 15:04:05.00",  // 2 decimals
		"2006-01-02 15:04:05.000", // 3 decimals
	}

	var lastErr error
	for _, layout := range layouts {
		t, err := time.Parse(layout, ts)
		if err == nil {
			return t, nil
		}
		lastErr = err
	}

	return time.Time{}, lastErr
}

func (db *db) openDatabase() error {
	database, openingError := sql.Open("sqlite3", "processed.db")

	if openingError != nil {
		return openingError
	}

	db.database = database
	return nil
}

func (db *db) getData() ([]string, error) {
	var timeStamps []string

	sqlQuery, _ := loadSQLFile("sql\\getData.sql")
	rows, queryError := db.database.Query(sqlQuery)

	if queryError != nil {
		return nil, queryError
	}
	defer rows.Close()

	for rows.Next() {
		var timeStamp, fileName string
		scanError := rows.Scan(&timeStamp, &fileName)
		if scanError != nil {
			return nil, scanError
		}
		timeStamps = append(timeStamps, timeStamp)
	}

	if rowError := rows.Err(); rowError != nil {
		return nil, rowError
	}

	return timeStamps, nil
}

func loadSQLFile(filePath string) (string, error) {

	content, err := os.ReadFile(filePath)
	if err != nil {
		return "", err
	}

	return string(content), nil
}
