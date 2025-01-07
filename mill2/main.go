package main

import (
	"database/sql"
	"fmt"
	_ "github.com/mattn/go-sqlite3"
	"os"
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

	timeStamps, queryError := dataBase.getData()
	if queryError != nil {
		fmt.Printf("Error querying database: %v\n", queryError)
		return
	}

	fmt.Println("Timestamps:", len(timeStamps))

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
		// fmt.Printf("TimeStamp: %s, Filename: %s\n", timeStamp, fileName)
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
