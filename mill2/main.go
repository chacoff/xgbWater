package main

import (
	"database/sql"
	"fmt"
	_ "github.com/mattn/go-sqlite3"
)

type db struct {
	database *sql.DB
}

var dataBase db = db{}

func (db *db) openDatabase() error {

	database, openingError := sql.Open("sqlite3", "processed.db")

	if openingError != nil {
		return openingError
	}

	db.database = database
	return nil
}

func main() {
	dataBase = db{}
	openError := dataBase.openDatabase()
	if openError != nil {
		return
	}

	timeStamps, queryError := dataBase.getData()
	if queryError != nil {
		fmt.Printf("Error querying database: %v\n", queryError)
		return
	}

	fmt.Println("Timestamps:", len(timeStamps))

}

func (db *db) getData() ([]string, error) {
	var timeStamps []string

	rows, queryError := db.database.Query("SELECT Timestamp, Filename FROM Measures WHERE Timestamp < '2024-12-20 15:22:00,000'")

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
		// fmt.Printf("TimeStamp: %s, Filename: %s\n", timeStamp, fileName)
	}

	if rowError := rows.Err(); rowError != nil {
		return nil, rowError
	}

	return timeStamps, nil
}

//WITH FilteredMeasures AS (
//SELECT Timestamp, Filename
//FROM Measures
//WHERE Timestamp < '2024-12-20 15:22:00,000' AND Filename = 'Pass 3'
//)
//
//SELECT MIN(TimeStamp) AS TimeStamp, Filename
//FROM FilteredMeasures
//UNION ALL
//SELECT MAX(TimeStamp) AS TimeStamp, Filename
//FROM FilteredMeasures;
